from __future__ import annotations
"""
TradingAI.py
Cerebro autonomo del bot: carga el modelo entrenado, recibe barras en vivo,
consulta la KnowledgeBase para contexto de mercado y emite ordenes
validadas por el RiskManager.

CAMBIO v2: KnowledgeBase integrada.
La IA_BackTests ahora consulta el VectorStore en cada decision para obtener contexto
de los libros, audios y videos que ya has ingresado. El contexto puede:
    1. BLOQUEAR una orden si el conocimiento contradice fuertemente la accion
    2. AJUSTAR el umbral de confianza segun el contexto de mercado
    3. REGISTRAR el contexto que influyó en cada decision para auditoria
"""
import threading
from typing import Optional
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from src.brain.FeatureEngineering import FeatureEngineer
from src.risk.RiskManager import RiskManager, RiskConfig
from src.brain.TradingEnvironment import TradingEnvironment
from config import settings as IBKR_SETTINGS


class KnowledgeFilter:
    """
    Capa de filtro que consulta el VectorStore y decide si el contexto
    del conocimiento almacenado apoya, contradice o es neutral respecto
    a la accion que el modelo PPO quiere ejecutar.

    Carga lazy: solo importa chromadb/sentence-transformers la primera vez
    que se consulta, para no ralentizar el arranque del bot.
    """

    # Palabras clave que el VectorStore debe devolver para que el contexto
    # se considere BULLISH, BEARISH o NEUTRAL respecto a la accion.
    BULLISH_KEYWORDS = {
        "compra", "buy", "alcista", "soporte", "ruptura", "momentum",
        "tendencia alcista", "breakout", "acumulacion", "demanda",
        "bull", "long", "entrada", "señal de compra",
    }
    BEARISH_KEYWORDS = {
        "venta", "sell", "bajista", "resistencia", "caida", "reversal",
        "tendencia bajista", "distribucion", "oferta",
        "bear", "short", "salida", "señal de venta",
    }

    def __init__(self, collection: str = "trading_knowledge", enabled: bool = True):
        self.enabled    = enabled
        self.collection = collection
        self._store     = None          # Carga lazy
        self._available = None          # None = no chequeado aún

    def _try_load(self) -> bool:
        """Intenta cargar el VectorStore. Retorna True si lo logró."""
        if self._available is not None:
            return self._available
        try:
            from src.knowledge.VectorStore import VectorStore
            self._store     = VectorStore(collection=self.collection)
            self._available = self._store.count() > 0
            if self._available:
                print(f"[KnowledgeFilter] ✓ KnowledgeBase cargada  "
                      f"({self._store.count()} chunks disponibles)")
            else:
                print("[KnowledgeFilter] KnowledgeBase vacía — "
                      "ingiere contenido con KnowledgeIngestor")
        except Exception as e:
            print(f"[KnowledgeFilter] No disponible: {e}")
            self._available = False
        return self._available

    def consult(
        self,
        action_name:   str,    # "BUY" | "SELL" | "HOLD"
        current_price: float,
        rsi:           float,
        atr_pct:       float,
        symbol:        str,
    ) -> dict:
        """
        Consulta la KnowledgeBase con el contexto actual del mercado.

        Retorna:
            {
              "score":    float,   # -1.0 (muy bearish) … +1.0 (muy bullish)
              "veto":     bool,    # True = el conocimiento contradice fuertemente la accion
              "context":  str,     # Fragmentos relevantes para el log
              "reason":   str,     # Explicación legible del resultado
            }
        """
        default = {"score": 0.0, "veto": False, "context": "", "reason": "KB no disponible"}

        if not self.enabled or not self._try_load():
            return default

        # Construir la consulta con el contexto actual
        rsi_str  = "sobrecompra" if rsi > 0.7 else ("sobreventa" if rsi < 0.3 else "neutral")
        atr_str  = "alta volatilidad" if atr_pct > 0.02 else "baja volatilidad"
        query    = (
            f"{symbol} {action_name.lower()} accion precio ${current_price:.2f} "
            f"RSI {rsi_str} {atr_str} estrategia entrada señal"
        )

        try:
            results = self._store.search(query, k=5, min_score=0.25)
        except Exception as e:
            print(f"[KnowledgeFilter] Error consultando KB: {e}")
            return default

        if not results:
            return {**default, "reason": "Sin resultados relevantes en KB"}

        # Analizar el sentimiento del contexto devuelto
        combined_text = " ".join(r["text"].lower() for r in results)
        bullish_hits  = sum(1 for w in self.BULLISH_KEYWORDS if w in combined_text)
        bearish_hits  = sum(1 for w in self.BEARISH_KEYWORDS if w in combined_text)
        total_hits    = bullish_hits + bearish_hits

        if total_hits == 0:
            score = 0.0
        else:
            score = (bullish_hits - bearish_hits) / total_hits   # -1.0 … +1.0

        # ¿El conocimiento contradice la accion?
        # BUY con contexto muy bearish (score < -0.5) → veto
        # SELL con contexto muy bullish (score > +0.5) → veto
        veto = False
        if action_name == "BUY"  and score < -0.5:
            veto = True
        if action_name == "SELL" and score > +0.5:
            veto = True

        # Construir contexto legible para el log
        context_lines = [
            f"[{r['source'].split('/')[-1][:30]} | {r['score']:.2f}] "
            f"{r['text'][:120]}..."
            for r in results[:3]
        ]
        context_str = "\n  ".join(context_lines)

        reason = (
            f"KB score={score:+.2f} "
            f"(bullish_hits={bullish_hits}, bearish_hits={bearish_hits}) "
            f"| {'VETO' if veto else 'OK'}"
        )

        return {
            "score":   round(score, 3),
            "veto":    veto,
            "context": context_str,
            "reason":  reason,
        }


class TradingAI:
    WARMUP_BARS = 60      # Barras minimas antes de operar

    def __init__(
        self,
        symbol:               str,
        risk_config:          Optional[RiskConfig] = None,
        confidence_threshold: float = 0.60,
        use_knowledge_base:   bool  = True,    # ← activa/desactiva la KB
        kb_veto_enabled:      bool  = True,    # ← si False, KB solo informa, no bloquea
    ):
        self.symbol               = symbol
        self.fe                   = FeatureEngineer()
        self.risk                 = RiskManager(config=risk_config)
        self.confidence_threshold = confidence_threshold
        self.model: Optional[PPO]              = None
        self.vec_normalize: Optional[VecNormalize] = None

        self._bar_buffer: list[dict] = []
        self._lock       = threading.Lock()
        self._order_cb   = None

        # KnowledgeBase filter
        self.kb_filter   = KnowledgeFilter(enabled=use_knowledge_base)
        self.kb_veto     = kb_veto_enabled

        self._action_names = {0: "HOLD", 1: "BUY", 2: "SELL"}

    # ── Setup ─────────────────────────────────────────────────────────────────

    def load(self) -> "TradingAI":
        """Carga el modelo PPO y el normalizador desde disco."""
        model_path = IBKR_SETTINGS.MODELS_DIR / self.symbol / "best_model"
        norm_path  = IBKR_SETTINGS.MODELS_DIR / self.symbol / "vec_normalize.pkl"

        if not model_path.with_suffix(".zip").exists():
            raise FileNotFoundError(
                f"[TradingAI] Modelo no encontrado: {model_path}\n"
                f"Ejecuta ModelTrainer().train(df) primero."
            )

        self.model = PPO.load(str(model_path))

        if norm_path.exists():
            dummy_env          = DummyVecEnv([lambda: TradingEnvironment(pd.DataFrame())])
            self.vec_normalize = VecNormalize.load(str(norm_path), dummy_env)
            self.vec_normalize.training = False

        print(f"[TradingAI] Modelo cargado para {self.symbol}")

        # Pre-cargar KB (para que no haya latencia en la primera barra)
        self.kb_filter._try_load()
        return self

    def set_order_callback(self, callback):
        self._order_cb = callback

    # ── Ingesta de datos en vivo ──────────────────────────────────────────────

    def on_new_bar(self, bar) -> None:
        with self._lock:
            self._bar_buffer.append({
                "open":   bar.open,
                "high":   bar.high,
                "low":    bar.low,
                "close":  bar.close,
                "volume": bar.volume,
            })

            if len(self._bar_buffer) < self.WARMUP_BARS:
                print(f"[TradingAI] Calentando ({len(self._bar_buffer)}/{self.WARMUP_BARS})...")
                return

            self._decide()

    # ── Decision ──────────────────────────────────────────────────────────────

    def _decide(self) -> None:
        """
        Pipeline completo de decision:
            PPO predict → KB consult → RiskManager → emit order
        """
        df_raw  = pd.DataFrame(self._bar_buffer[-200:])
        df_feat = self.fe.transform(df_raw)

        if len(df_feat) < TradingEnvironment(pd.DataFrame()).window:
            return

        obs = self._build_observation(df_feat)

        # ── 1. Prediccion del modelo PPO ──────────────────────────────────────
        raw_action, _ = self.model.predict(obs, deterministic=True)
        action        = int(raw_action)
        confidence    = self._estimate_confidence(obs)
        action_name   = self._action_names.get(action, "UNKNOWN")

        current_price = float(df_feat["close"].iloc[-1])
        atr_norm      = float(df_feat.get("atr_norm", pd.Series([0])).iloc[-1])
        atr_abs       = atr_norm * current_price
        rsi           = float(df_feat.get("rsi", pd.Series([0.5])).iloc[-1])

        print(
            f"\n[TradingAI] {self.symbol}  "
            f"Accion: {action_name}  |  "
            f"Confianza: {confidence:.2f}  |  "
            f"Precio: ${current_price:.2f}"
        )

        # ── 2. Consulta a la KnowledgeBase ────────────────────────────────────
        kb_result = self.kb_filter.consult(
            action_name   = action_name,
            current_price = current_price,
            rsi           = rsi,
            atr_pct       = atr_norm,
            symbol        = self.symbol,
        )

        if kb_result["context"]:
            print(f"[KnowledgeBase] {kb_result['reason']}")
            print(f"  Contexto relevante:\n  {kb_result['context']}")

        # Veto de la KnowledgeBase
        if self.kb_veto and kb_result["veto"] and action in (1, 2):
            print(
                f"[TradingAI] ⛔ Orden VETADA por KnowledgeBase  "
                f"(score={kb_result['score']:+.2f}  accion={action_name})"
            )
            return

        # Ajuste de confianza por KB: si el conocimiento respalda la accion,
        # bajamos levemente el umbral; si la contradice (sin veto), lo subimos.
        effective_confidence = confidence
        if action in (1, 2) and kb_result["score"] != 0.0:
            # score +1 → umbral -0.05  |  score -1 → umbral +0.05
            adjustment = -kb_result["score"] * 0.05
            effective_confidence = confidence + adjustment
            if adjustment != 0:
                print(
                    f"[TradingAI] Confianza ajustada por KB: "
                    f"{confidence:.3f} → {effective_confidence:.3f} "
                    f"(KB score {kb_result['score']:+.2f})"
                )

        # ── 3. RiskManager ────────────────────────────────────────────────────
        allowed, reason = self.risk.check(action, effective_confidence, current_price, atr_abs)
        if not allowed:
            print(f"[TradingAI] Bloqueado por RiskManager: {reason}")
            return

        # ── 4. Emitir orden ───────────────────────────────────────────────────
        if action in (1, 2) and self._order_cb:
            size          = self.risk.position_size(current_price)
            profit_target = self.risk.dynamic_take_profit(current_price, atr_abs)
            stop_loss     = self.risk.dynamic_stop_loss(current_price, atr_abs)

            print(
                f"[TradingAI] ✅ Ejecutando {action_name} × {size} @ ${current_price:.2f}  "
                f"TP=${profit_target:.2f}  SL=${stop_loss:.2f}"
            )

            self._order_cb(
                action        = "BUY" if action == 1 else "SELL",
                quantity      = size,
                profit_target = profit_target,
                stop_loss     = stop_loss,
            )

    # ── Utilidades ────────────────────────────────────────────────────────────

    def _build_observation(self, df_feat: pd.DataFrame) -> np.ndarray:
        env = TradingEnvironment(df_feat)
        env.current_step = len(df_feat) - 1
        obs = env._get_observation()
        if self.vec_normalize:
            obs = self.vec_normalize.normalize_obs(obs[np.newaxis])[0]
        return obs

    def _estimate_confidence(self, obs: np.ndarray) -> float:
        try:
            import torch
            obs_tensor = torch.FloatTensor(obs[np.newaxis])
            with torch.no_grad():
                dist  = self.model.policy.get_distribution(obs_tensor)
                probs = dist.distribution.probs.numpy()[0]
            entropy     = -np.sum(probs * np.log(probs + 1e-9))
            max_entropy = np.log(len(probs))
            return float(1.0 - (entropy / max_entropy))
        except Exception:
            return 0.75

    def get_status(self) -> dict:
        kb_chunks = 0
        if self.kb_filter._available:
            try:
                kb_chunks = self.kb_filter._store.count()
            except Exception:
                pass
        return {
            "symbol":        self.symbol,
            "bars_loaded":   len(self._bar_buffer),
            "risk":          self.risk.get_status(),
            "model_ready":   self.model is not None,
            "kb_enabled":    self.kb_filter.enabled,
            "kb_chunks":     kb_chunks,
            "kb_veto_on":    self.kb_veto,
        }


"""
    Agente de trading autonomo basado en PPO + RiskManager + KnowledgeBase.

    Flujo en vivo:
        1. Recibe nueva barra via on_new_bar()
        2. Calcula features con FeatureEngineer
        3. Pide accion al modelo (PPO.predict)
        4. Consulta KnowledgeBase con el contexto actual  ← NUEVO
        5. Valida con RiskManager
        6. Emite senal via callback → Portfolio

    Uso:
        ai = TradingAI(symbol="AAPL")
        ai.load()
        ai.set_order_callback(portfolio.place_bracket_order)
        market_data.set_bar_close_callback(ai.on_new_bar)
"""
