"""
trade_reviewer.py
─────────────────────────────────────────────────────────────────────────────
Loads HTF_verification_timeline.csv and lets you step through every trade.

Snapshot toggle (button or Shift+T):
  HTF — signal timeframe, CONTEXT_BARS before signal candle → exit candle.
        TF is auto-detected from (LTF_Exec_Time - HTF_Signal_Time) in the CSV.
  LTF — always 1min, CONTEXT_BARS before entry candle → exit candle.
  Navigating to another trade always returns to replay mode.

Replay controls (Play / +1 Bar / Speed) work in normal mode as before.
The tf_switch still controls the replay TF as usual; it does NOT affect
the HTF snapshot TF (that is fixed by the data).
"""

import os
import sys
import time
import threading
import pandas as pd
import http.server
import urllib.parse
from lightweight_charts import Chart

# ── Paths & constants ─────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
DATA_DIR       = "../data"
CONTEXT_BARS   = 2000
MAX_CHART_BARS = 25000
TIMEFRAMES     = ["1min", "5min", "15min", "30min", "1h", "4h", "1d"]
TF_SECS        = {
    "1min": 60, "5min": 300, "15min": 900, "30min": 1800,
    "1h": 3600, "4h": 14400, "1d": 86400,
}
SPEED_OPTIONS  = ("1x", "5x", "10x", "30x", "60x")
SPEED_MAP      = {"1x": 1, "5x": 5, "10x": 10, "30x": 30, "60x": 60}
TZ_MAP         = {
    "PST": "America/Los_Angeles",
}

# Snapshot modes — replay is the default, toggle flips between htf and ltf
_MODE_REPLAY = "replay"
_MODE_HTF    = "htf"
_MODE_LTF    = "ltf"

# Minutes → TF label (for auto-detected signal TF)
_MINS_TO_TF = {
    1: "1min", 5: "5min", 15: "15min", 30: "30min",
    60: "1h", 240: "4h", 1440: "1d",
}


# ── CSV loader ────────────────────────────────────────────────────────────────
def load_timeline(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    df["_signal_utc"] = pd.to_datetime(
        df["HTF_Signal_Time"], utc=True
    ).dt.tz_localize(None)

    for col in ("LTF_Exec_Time", "Exit_Time"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce").dt.tz_localize(None)

    df = df.sort_values("_signal_utc").reset_index(drop=True)
    return df


def detect_htf_tf(timeline: pd.DataFrame) -> str:
    """
    Infer the signal timeframe from the CSV itself.
    LTF_Exec_Time is always HTF_Signal_Time + htf_mins, so the median
    difference across all rows reliably recovers the original htf_mins.
    Falls back to "5min" if detection fails.
    """
    try:
        sig  = timeline["_signal_utc"]
        exec_ = timeline["LTF_Exec_Time"]
        valid = exec_.notna() & sig.notna()
        diffs = (exec_[valid] - sig[valid]).dt.total_seconds() / 60
        median_mins = int(round(diffs.median()))
        tf = _MINS_TO_TF.get(median_mins)
        if tf:
            print(f"  Auto-detected HTF signal TF: {tf} ({median_mins} min)")
            return tf
    except Exception as e:
        print(f"  HTF TF detection failed ({e}), defaulting to 5min")
    return "5min"


# ── Local Web UI Bridge ───────────────────────────────────────────────────────
class NativeUIBridge(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/jump?date="):
            query    = urllib.parse.urlparse(self.path).query
            params   = urllib.parse.parse_qs(query)
            date_val = params.get("date", [""])[0]

            self.send_response(200)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(b"OK")

            if date_val and hasattr(self.server, "app_ref"):
                formatted_date = date_val.replace("T", " ")
                threading.Thread(
                    target=self.server.app_ref._load_for_date,
                    args=(formatted_date,),
                    daemon=True,
                ).start()
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass


# ── Parquet loader & Timezone Shifter ─────────────────────────────────────────
_CACHE: dict[str, pd.DataFrame] = {}

def load_parquet(tf: str) -> pd.DataFrame:
    if tf in _CACHE:
        return _CACHE[tf]
    path = os.path.join(DATA_DIR, f"nq_{tf}.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path} — run python preprocess.py first.")

    df = pd.read_parquet(path, engine="pyarrow")
    df.index = pd.to_datetime(df.index).tz_localize(None).astype("datetime64[ns]")
    df.index.name = "time"
    df = df.sort_index()
    df = df[["open", "high", "low", "close", "volume"]].dropna()
    df["open"]   = df["open"].astype("float32")
    df["high"]   = df["high"].astype("float32")
    df["low"]    = df["low"].astype("float32")
    df["close"]  = df["close"].astype("float32")
    df["volume"] = df["volume"].astype("int32")

    _CACHE[tf] = df
    return df


def get_tz_df(tf: str, tz_label: str) -> pd.DataFrame:
    df = load_parquet(tf).copy()
    if tz_label != "UTC":
        actual_tz = TZ_MAP[tz_label]
        df.index  = (
            df.index.tz_localize("UTC")
            .tz_convert(actual_tz)
            .tz_localize(None)
        )
    return df


def _ts_to_tz(ts_utc_naive: pd.Timestamp, tz_label: str) -> pd.Timestamp:
    if tz_label == "UTC":
        return ts_utc_naive
    return (
        ts_utc_naive
        .tz_localize("UTC")
        .tz_convert(TZ_MAP[tz_label])
        .tz_localize(None)
    )


def find_start_idx(df: pd.DataFrame, date_str: str) -> int:
    try:
        target = pd.Timestamp(date_str)
    except Exception:
        target = df.index[-1]
    pos = int(df.index.searchsorted(target))
    pos = min(pos, len(df) - 1)
    pos = max(pos, CONTEXT_BARS)
    return pos


# ── Topbar info helper ────────────────────────────────────────────────────────
def _build_trade_info(row: pd.Series, signal_ts_display: str) -> str:
    sig       = row.get("HTF_Signal",       "—")
    direction = row.get("Direction_Traded", "—")
    outcome   = str(row.get("Outcome", "—")).upper()
    entry     = row.get("Entry_Price", "—")
    exit_px   = row.get("Exit_Price",  "—")
    mfe       = row.get("Window_MFE",  "—")
    mae       = row.get("Window_MAE",  "—")

    def fmt(v):
        try:
            return f"{float(v):.1f}"
        except Exception:
            return str(v)

    icon = "✅" if outcome == "WIN" else ("❌" if "LOSS" in outcome else "⏱")

    return (
        f"{signal_ts_display}  |  {sig}  {direction}  |  "
        f"In {fmt(entry)}  Out {fmt(exit_px)}  |  "
        f"MFE {fmt(mfe)}  MAE {fmt(mae)}  |  {icon} {outcome}"
    )


# ── App ───────────────────────────────────────────────────────────────────────
class TradeReviewApp:
    def __init__(self, csv_path: str):
        # ── HTTP bridge ──────────────────────────────────────────────────────
        self.server = http.server.HTTPServer(("127.0.0.1", 0), NativeUIBridge)
        self.server.app_ref = self
        self.server_port    = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

        # ── Load timeline & detect HTF TF ─────────────────────────────────────
        self.timeline     = load_timeline(csv_path)
        self.total_trades = len(self.timeline)
        self.htf_tf       = detect_htf_tf(self.timeline)   # fixed for this CSV
        print(f"Loaded {self.total_trades} trades from {csv_path}")

        # ── Chart setup ───────────────────────────────────────────────────────
        self.chart = Chart(title="NQ Trade Reviewer", maximize=True, toolbox=True)
        self.chart.layout(background_color="#0d1117", text_color="#e6edf3", font_size=12, font_family="monospace")
        self.chart.candle_style(
            up_color="#3fb950", down_color="#f85149",
            wick_up_color="#3fb950", wick_down_color="#f85149",
            border_visible=False,
        )
        self.chart.volume_config(scale_margin_top=0.82, scale_margin_bottom=0.0)
        self.chart.grid(vert_enabled=True, horz_enabled=True, color="rgba(255,255,255,0.04)")
        self.chart.crosshair(mode="normal", vert_style="dashed", horz_style="dashed",
                             vert_color="rgba(200,200,200,0.4)", horz_color="rgba(200,200,200,0.4)")
        self.chart.legend(visible=True, ohlc=True, percent=False, color_based_on_candle=True, font_size=13)
        self.chart.time_scale(right_offset=8, time_visible=True, seconds_visible=False)

        # ── State ─────────────────────────────────────────────────────────────
        self._lock      = threading.Lock()
        self.full_df    = pd.DataFrame()
        self.replay_idx = 0
        self.is_playing = False
        self.speed      = 1
        self.tf         = "1min"          # replay TF (controlled by tf_switch)
        self.tz         = "PST"
        self.elapsed    = 0.0
        self.snap_mode  = _MODE_REPLAY
        self.trade_idx  = 0

        # ── Top bar ───────────────────────────────────────────────────────────
        # Trade navigation
        self.chart.topbar.button("btn_prev",       "◀",                   separator=False, func=self._on_prev)
        self.chart.topbar.textbox("trade_counter", f"1 / {self.total_trades}")
        self.chart.topbar.button("btn_next_trade", "▶",                   separator=False, func=self._on_next_trade)

        # TF & TZ (tf_switch controls replay TF only; HTF snapshot uses auto-detected TF)
        self.chart.topbar.switcher("tf_switch", tuple(TIMEFRAMES), default="1min", func=self._on_tf)

        # Snapshot toggle — two states only: HTF ↔ LTF
        # Label always shows the CURRENT view so the user knows where they are.
        # Clicking switches to the other view.
        self.chart.topbar.button(
            "btn_snap",
            f"📊 HTF ({self.htf_tf})",
            separator=True,
            func=self._on_snap_toggle,
        )

        # Replay controls
        self.chart.topbar.button("btn_play",  "▶  Play",   separator=True,  func=self._on_play)
        self.chart.topbar.button("btn_step",  "⏭  +1 Bar", separator=False, func=self._on_step)
        self.chart.topbar.switcher("speed_switch", SPEED_OPTIONS, default="1x", func=self._on_speed)
        # Trade info
        self.chart.topbar.textbox("trade_info", "—")

        # Hotkeys
        self.chart.hotkey("shift", "P", self._on_play)
        self.chart.hotkey(None,    39,  self._on_step)           # →       step bar
        self.chart.hotkey("shift", 39,  self._on_next_trade)     # Shift+→ next trade
        self.chart.hotkey("shift", 37,  self._on_prev)           # Shift+← prev trade
        self.chart.hotkey("shift", "T", self._on_snap_toggle)    # Shift+T toggle snapshot

        # ── Pre-load & start ──────────────────────────────────────────────────
        print("Pre-loading 1min data...", end=" ", flush=True)
        self.full_df = get_tz_df("1min", "PST") # <-- changed to PST
        print(f"{len(self.full_df):,} rows")

        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

        self._go_to_trade(self.total_trades - 1)

    # ─── SNAPSHOT HELPERS ─────────────────────────────────────────────────────

    def _exit_ts_in_tz(self, row: pd.Series) -> pd.Timestamp | None:
        exit_utc = row.get("Exit_Time")
        if pd.isna(exit_utc) or exit_utc is None:
            return None
        return _ts_to_tz(pd.Timestamp(exit_utc), self.tz)

    def _slice_to_exit(self, df: pd.DataFrame, context_anchor_pos: int,
                        exit_ts: pd.Timestamp | None, fallback_bars: int = 10) -> pd.DataFrame:
        """
        Return a slice of df from CONTEXT_BARS before context_anchor_pos
        through to (and including) the exit candle. Shared by both snapshot modes.
        """
        anchor = min(context_anchor_pos, len(df) - 1)

        if exit_ts is not None:
            exit_pos = int(df.index.searchsorted(exit_ts, side="right")) - 1
            exit_pos = max(anchor, min(exit_pos, len(df) - 1))
            end = min(exit_pos + 1, len(df))
        else:
            end = min(anchor + fallback_bars, len(df))

        start = max(0, min(anchor - CONTEXT_BARS, end - 1))
        return df.iloc[start:end].copy(), end

    def _show_htf_snapshot(self, row: pd.Series):
        """
        HTF view: auto-detected signal TF, signal candle through exit.
        The tf_switch is NOT used here — the TF is fixed by the data.
        """
        signal_ts = _ts_to_tz(row["_signal_utc"], self.tz)
        exit_ts   = self._exit_ts_in_tz(row)

        df = get_tz_df(self.htf_tf, self.tz)

        sig_pos = int(df.index.searchsorted(signal_ts))
        sig_pos = min(sig_pos, len(df) - 1)

        snapshot, end_idx = self._slice_to_exit(df, sig_pos, exit_ts, fallback_bars=10)
        self.chart.set(snapshot)

        with self._lock:
            self.full_df    = df
            self.replay_idx = end_idx

        self.chart.watermark(
            f"HTF  ({self.htf_tf})",
            font_size=14, color="rgba(100,200,255,0.25)",
        )

    def _show_ltf_snapshot(self, row: pd.Series):
        """
        LTF view: always 1min, entry candle through exit.
        """
        exec_utc = row.get("LTF_Exec_Time")
        exec_ts  = (
            _ts_to_tz(pd.Timestamp(exec_utc), self.tz)
            if (exec_utc is not None and not pd.isna(exec_utc))
            else _ts_to_tz(row["_signal_utc"], self.tz)
        )
        exit_ts = self._exit_ts_in_tz(row)

        df_1m = get_tz_df("1min", self.tz)

        entry_pos = int(df_1m.index.searchsorted(exec_ts))
        entry_pos = min(entry_pos, len(df_1m) - 1)

        snapshot, end_idx = self._slice_to_exit(df_1m, entry_pos, exit_ts, fallback_bars=30)
        self.chart.set(snapshot)

        with self._lock:
            self.full_df    = df_1m
            self.replay_idx = end_idx

        self.chart.watermark(
            "LTF  (1min)",
            font_size=14, color="rgba(255,200,80,0.25)",
        )

    def _snap_button_label(self):
        if self.snap_mode == _MODE_HTF:
            return f"📊 HTF ({self.htf_tf})"
        return "📈 LTF (1min)"
    
    def _on_snap_toggle(self, chart):
        """
        Two-state toggle between HTF and LTF snapshot.
        If currently in replay mode, first press enters HTF.
        HTF → LTF, LTF → HTF. No third "back to replay" state —
        navigation (◀ / ▶) is the natural way back to replay.
        """
        row = self.timeline.iloc[self.trade_idx]

        # Stop any active replay before switching view
        with self._lock:
            self.is_playing = False
            self.elapsed    = 0.0
        self.chart.topbar["btn_play"].set("▶  Play")

        if self.snap_mode == _MODE_LTF:
            self.snap_mode = _MODE_HTF
            self._show_htf_snapshot(row)
        else:
            # replay or htf → go to LTF
            self.snap_mode = _MODE_LTF
            self._show_ltf_snapshot(row)

        self.chart.topbar["btn_snap"].set(self._snap_button_label())

    # ─── TRADE NAVIGATION ─────────────────────────────────────────────────────

    def _go_to_trade(self, idx: int):
        """Jump to trade in replay mode. Always resets snapshot state."""
        idx = idx % self.total_trades
        self.trade_idx = idx
        self.snap_mode = _MODE_REPLAY

        row       = self.timeline.iloc[idx]
        row = self.timeline.iloc[idx]

        with self._lock:
            self.is_playing = False
            self.elapsed = 0.0

        self.chart.topbar["btn_play"].set("▶  Play")

        self.snap_mode = _MODE_HTF
        self._show_htf_snapshot(row)

        self.chart.topbar["trade_counter"].set(
            f"{idx + 1} / {self.total_trades}"
        )

        signal_ts_disp = _ts_to_tz(row["_signal_utc"], self.tz)

        self.chart.topbar["trade_info"].set(
            _build_trade_info(
                row,
                signal_ts_disp.strftime("%A, %B %d, %Y %H:%M")
            )
        )

        self.chart.topbar["btn_snap"].set(
            self._snap_button_label()
        )

        print(
            f"  → Trade {idx + 1}/{self.total_trades}: "
            f"{row.get('HTF_Signal','?')} "
            f"{row.get('Direction_Traded','?')} "
            f"@ {signal_ts_disp.strftime('%Y-%m-%d %H:%M')} "
            f"{self.tz} "
            f"[{row.get('Outcome','?')}]"
        )

    def _on_prev(self, chart):
        self._go_to_trade(self.trade_idx - 1)

    def _on_next_trade(self, chart):
        self._go_to_trade(self.trade_idx + 1)

    # ─── TF / TZ ──────────────────────────────────────────────────────────────

    def _on_tf(self, chart):
        with self._lock:
            self.tf = self.chart.topbar["tf_switch"].value
        # TF switch only affects replay mode; re-jump to refresh
        if self.snap_mode == _MODE_REPLAY:
            self._go_to_trade(self.trade_idx)
        # In snapshot modes the HTF TF is fixed and LTF is always 1min —
        # so a tf_switch change has no effect on the current snapshot view.
        # The new tf value will take effect next time the user returns to replay.

    # ─── REPLAY ───────────────────────────────────────────────────────────────

    def _load_for_date(self, date_str: str):
        with self._lock:
            was_playing     = self.is_playing
            self.is_playing = False

        self.chart.topbar["btn_play"].set("▶  Play")

        df  = get_tz_df(self.tf, self.tz)
        idx = find_start_idx(df, date_str)

        history_end   = min(idx + 1, len(df))
        history_start = max(0, history_end - MAX_CHART_BARS)
        self.chart.set(df.iloc[history_start:history_end].copy())

        with self._lock:
            self.full_df    = df
            self.replay_idx = history_end
            self.elapsed    = 0.0

        self._update_timer(TF_SECS[self.tf])

        if was_playing:
            with self._lock:
                self.is_playing = True
            self.chart.topbar["btn_play"].set("⏸  Pause")

    def _push_bar(self):
        with self._lock:
            if self.replay_idx >= len(self.full_df):
                self.is_playing = False
                self.chart.topbar["btn_play"].set("▶  Play")
                self.chart.watermark("END OF DATA", font_size=22, color="rgba(200,80,80,0.6)")
                return

            row = self.full_df.iloc[self.replay_idx]
            bar = pd.Series({
                "time":   row.name,
                "open":   float(row["open"]),
                "high":   float(row["high"]),
                "low":    float(row["low"]),
                "close":  float(row["close"]),
                "volume": int(row["volume"]),
            })
            self.replay_idx += 1

        self.chart.update(bar)

    def _loop(self):
        TICK = 0.05
        while self._running:
            with self._lock:
                playing = self.is_playing
                speed   = self.speed
                tf_secs = TF_SECS[self.tf]

            if not playing:
                time.sleep(TICK)
                continue

            with self._lock:
                self.elapsed += TICK * speed

            while True:
                with self._lock:
                    fire = self.elapsed >= tf_secs
                if not fire:
                    break
                self._push_bar()
                with self._lock:
                    self.elapsed -= tf_secs

            with self._lock:
                remaining_sim = tf_secs - self.elapsed

            self._update_timer(remaining_sim / max(speed, 1))
            time.sleep(TICK)

    # ─── CALLBACKS ────────────────────────────────────────────────────────────

    def _on_play(self, chart):
        with self._lock:
            if self.replay_idx >= len(self.full_df):
                return
            self.is_playing = not self.is_playing
            playing = self.is_playing
            if playing:
                self.elapsed = 0.0
        self.chart.topbar["btn_play"].set("⏸  Pause" if playing else "▶  Play")

    def _on_step(self, chart):
        with self._lock:
            if len(self.full_df) == 0:
                return
            self.elapsed = 0.0
        self._push_bar()
        with self._lock:
            tf_secs = TF_SECS[self.tf]
        self._update_timer(tf_secs)

    def _on_speed(self, chart):
        val = self.chart.topbar["speed_switch"].value
        with self._lock:
            self.speed = SPEED_MAP.get(val, 1)

    # ─── TIMER ────────────────────────────────────────────────────────────────

    def _update_timer(self, wall_seconds: float):
        s = max(0.0, wall_seconds)
        h, rem = divmod(int(s), 3600)
        m, sec = divmod(rem, 60)
        label  = f"{h:02d}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"
        self.chart.watermark(f"Next bar: {label}", font_size=18, color="rgba(150,150,150,0.35)")

    # ─── RUN ──────────────────────────────────────────────────────────────────

    def run(self):
        self.chart.show(block=True)
        self._running = False


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "HTF_verification_timeline_15.csv"
    if not os.path.exists(csv_path):
        print(f"ERROR: CSV not found: {csv_path}")
        sys.exit(1)

    app = TradeReviewApp(csv_path)
    app.run()