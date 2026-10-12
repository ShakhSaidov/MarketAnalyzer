import os
import random
import time
import threading
import pandas as pd
from lightweight_charts import Chart

# ── Paths & constants ─────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CONTEXT_BARS = 2000
MAX_CHART_BARS = 25000

TIMEFRAMES = ["1min", "5min", "15min", "30min", "1h", "4h", "1d"]
TF_SECS = {
    "1min": 60, "5min": 300, "15min": 900, "30min": 1800,
    "1h": 3600, "4h": 14400, "1d": 86400,
}
SPEED_OPTIONS = ("1x", "5x", "10x", "30x", "60x")
SPEED_MAP = {"1x": 1, "5x": 5, "10x": 10, "30x": 30, "60x": 60}

# Mapping UI labels to standard IANA timezone database names (Handles DST Automatically)
TZ_MAP = {
    "UTC": "UTC",
    "EST": "America/New_York",
    "PST": "America/Los_Angeles"
}
ET_TZ = "America/New_York"
SESSION_OPEN_HHMM = "09:30"  # regular session open, used as the anchor for date navigation

# The key used internally to store/retrieve drawings in toolbox.drawings dict.
# Since this is always NQ, we use a fixed key so drawings survive all TF/date switches.
DRAWINGS_KEY = "NQ"


# ── Parquet loader & Timezone Shifter ─────────────────────────────────────────
_CACHE: dict[str, pd.DataFrame] = {}

def load_parquet(tf: str) -> pd.DataFrame:
    if tf in _CACHE: return _CACHE[tf]
    path = os.path.join(DATA_DIR, f"nq_{tf}.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path} — run python preprocess.py first.")

    df = pd.read_parquet(path, engine="pyarrow")
    # Base dataset is loaded as UTC Naive
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
    """Pulls from UTC cache, applies requested timezone, returns Naive DF for UI"""
    df = load_parquet(tf).copy()
    if tz_label != "UTC":
        actual_tz = TZ_MAP[tz_label]
        # Localize to UTC, shift to target TZ, strip timezone data for frontend UI
        df.index = df.index.tz_localize("UTC").tz_convert(actual_tz).tz_localize(None)
    return df

def compute_trading_days(df_1min_utc: pd.DataFrame) -> list:
    """
    Trading days = ET calendar dates that have a bar at exactly the regular
    session open (09:30 ET). Same convention used elsewhere in this project
    (trading-day detection via presence of a 09:30 ET bar).
    `df_1min_utc` must be the 1min dataframe with a tz-naive, UTC-valued index
    (i.e. whatever load_parquet/get_tz_df("1min", "UTC") returns).
    """
    et_index = df_1min_utc.index.tz_localize("UTC").tz_convert(ET_TZ)
    hh, mm = (int(x) for x in SESSION_OPEN_HHMM.split(":"))
    mask = (et_index.hour == hh) & (et_index.minute == mm)
    return sorted(set(et_index[mask].strftime("%Y-%m-%d")))


# ── App ───────────────────────────────────────────────────────────────────────
class ReplayApp:
    def __init__(self):
        self.chart = Chart(title="NQ Futures Replay", maximize=True, toolbox=True)
        self.chart.layout(background_color="#0d1117", text_color="#e6edf3", font_size=12, font_family="monospace")
        self.chart.candle_style(up_color="#3fb950", down_color="#f85149", wick_up_color="#3fb950", wick_down_color="#f85149", border_visible=False)
        self.chart.volume_config(scale_margin_top=0.82, scale_margin_bottom=0.0)
        self.chart.grid(vert_enabled=True, horz_enabled=True, color="rgba(255,255,255,0.04)")
        self.chart.crosshair(mode="normal", vert_style="dashed", horz_style="dashed", vert_color="rgba(200,200,200,0.4)", horz_color="rgba(200,200,200,0.4)")
        self.chart.legend(visible=True, ohlc=True, percent=False, color_based_on_candle=True, font_size=13)
        self.chart.time_scale(right_offset=8, time_visible=True, seconds_visible=False)

        self._lock      = threading.Lock()
        self.full_df    = pd.DataFrame()
        self.replay_idx = 0
        self.is_playing = False
        self.speed      = 1
        self.tf         = "1min"
        self.tz         = "UTC"
        self.elapsed    = 0.0

        # ── Hindsight-free date navigation state ───────────────────────────────
        self.visited_dates: list[str] = []
        self.history_ptr: int = -1
        self._deck: list[str] = []
        
        # Single source of truth for the simulated clock (Timezone Aware UTC)
        self.current_sim_time_utc: pd.Timestamp = None

        # ── Drawing Persistence Setup ──────────────────────────────────────────
        self.chart.topbar.textbox("_symbol", DRAWINGS_KEY)
        self.chart.toolbox.save_drawings_under(self.chart.topbar["_symbol"])

        # ── Override drawing line width & hide Date on the X-Axis ─────────────
        self.chart.run_script(f"""
            setTimeout(() => {{
                const h = {self.chart.id};
                if (h && h.toolBox && h.toolBox._drawingTool) {{
                    const dt = h.toolBox._drawingTool;

                    const origAdd = dt.addNewDrawing.bind(dt);
                    dt.addNewDrawing = function(drawing) {{
                        if (drawing && drawing._options) drawing._options.width = 1;
                        origAdd(drawing);
                    }};

                    const origClick = dt._onClick.bind(dt);
                    dt._onClick = function(e) {{
                        origClick(e);
                        if (dt._activeDrawing && dt._activeDrawing._options) {{
                            dt._activeDrawing._options.width = 1;
                            dt._activeDrawing.requestUpdate && dt._activeDrawing.requestUpdate();
                        }}
                    }};
                }}
                
                // Hide Date from X-Axis tick marks and Crosshair (Show Time Only)
                if (h && h.chart) {{
                    h.chart.applyOptions({{
                        localization: {{
                            timeFormatter: (time) => {{
                                if (typeof time === 'object' || typeof time === 'string') return 'Daily';
                                const d = new Date(time * 1000);
                                const hh = d.getUTCHours().toString().padStart(2, '0');
                                const mm = d.getUTCMinutes().toString().padStart(2, '0');
                                return hh + ':' + mm;
                            }}
                        }},
                        timeScale: {{
                            tickMarkFormatter: (time, tickMarkType, locale) => {{
                                if (typeof time === 'object' || typeof time === 'string') return 'Daily';
                                const d = new Date(time * 1000);
                                const hh = d.getUTCHours().toString().padStart(2, '0');
                                const mm = d.getUTCMinutes().toString().padStart(2, '0');
                                return hh + ':' + mm;
                            }}
                        }}
                    }});
                }}
            }}, 800);
        """)

        # Weekday-only display — never reveals the actual date
        self.chart.topbar.textbox("date_display", "—")

        # Timeframe & Timezone switchers
        self.chart.topbar.switcher("tf_switch", tuple(TIMEFRAMES), default="1min", func=self._on_tf)
        self.chart.topbar.switcher("tz_switch", ("UTC", "EST", "PST"), default="UTC", func=self._on_tz)

        # Date navigation
        self.chart.topbar.button("btn_prev_date", "◀ Prev", separator=True, func=self._on_prev_date)
        self.chart.topbar.button("btn_next_date", "Next ▶", separator=False, func=self._on_next_date)

        self.chart.topbar.button("btn_play", "▶  Play", separator=True, func=self._on_play)
        self.chart.topbar.button("btn_next", "⏭  +1 Bar", separator=False, func=self._on_next)
        self.chart.topbar.switcher("speed_switch", SPEED_OPTIONS, default="1x", func=self._on_speed)
        self.chart.topbar.button("btn_clear", "✕ Clear Drawings", separator=True, func=self._on_clear_drawings)

        self.chart.hotkey("shift", "P", self._on_play)
        self.chart.hotkey(None, 39, self._on_next)

        print("Pre-loading 1min data...", end=" ", flush=True)
        self.full_df = get_tz_df("1min", "UTC")
        print(f"{len(self.full_df):,} rows")

        self.all_trading_days = compute_trading_days(self.full_df)
        if not self.all_trading_days:
            raise RuntimeError(
                "No trading days found (no 09:30 ET bars) in the 1min data — "
                "cannot start replay."
            )
        print(f"Found {len(self.all_trading_days):,} trading days available for practice.")

        # Land immediately on a random session
        self._advance_to_new_day()

        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    # ─── HINDSIGHT-FREE DATE NAVIGATION ───────────────────────────────────────
    def _draw_next_date(self) -> str:
        if not self._deck:
            self._deck = self.all_trading_days.copy()
            random.shuffle(self._deck)
            if self.visited_dates and len(self._deck) > 1 and self._deck[-1] == self.visited_dates[-1]:
                self._deck[-1], self._deck[-2] = self._deck[-2], self._deck[-1]
        return self._deck.pop()

    def _goto_trading_day(self, date_str_et: str):
        """Jumps strictly to the exact moment of the session open (09:30:00). 
        Calculates the highest allowed historical candle dynamically."""
        et_ts = pd.Timestamp(f"{date_str_et} {SESSION_OPEN_HHMM}:00", tz=ET_TZ)
        self.current_sim_time_utc = et_ts.tz_convert("UTC")
        self._load_for_sim_time(clear_drawings=True)

    def _advance_to_new_day(self):
        date_str = self._draw_next_date()
        self.visited_dates.append(date_str)
        self.history_ptr = len(self.visited_dates) - 1
        self._goto_trading_day(date_str)

    def _on_next_date(self, chart):
        with self._lock:
            at_frontier = self.history_ptr >= len(self.visited_dates) - 1
        if at_frontier:
            self._advance_to_new_day()
        else:
            with self._lock:
                self.history_ptr += 1
                date_str = self.visited_dates[self.history_ptr]
            self._goto_trading_day(date_str)

    def _on_prev_date(self, chart):
        with self._lock:
            if self.history_ptr <= 0:
                print("Already at the earliest visited session — can't go further back.")
                return
            self.history_ptr -= 1
            date_str = self.visited_dates[self.history_ptr]
        self._goto_trading_day(date_str)

    # ─── LOAD (DYNAMIC ANCHOR LOGIC) ──────────────────────────────────────────
    def _load_for_sim_time(self, clear_drawings: bool = False):
        with self._lock:
            was_playing     = self.is_playing
            self.is_playing = False

        self.chart.topbar["btn_play"].set("▶  Play")

        # Load data shifted to user's currently selected timezone
        df = get_tz_df(self.tf, self.tz)

        # 1. Figure out the naive timestamp representing our global clock for this TZ
        sim_time_tz_aware = self.current_sim_time_utc.tz_convert(TZ_MAP[self.tz])
        sim_time_naive = sim_time_tz_aware.tz_localize(None)

        # 2. To strictly prevent leaking incomplete candles, we only load candles 
        # whose (open timestamp + timeframe duration) <= current simulated clock.
        max_allowed_ts = sim_time_naive - pd.Timedelta(seconds=TF_SECS[self.tf])
        
        # 3. Find highest allowable index
        idx = df.index.searchsorted(max_allowed_ts, side='right') - 1

        if idx < 0:
            history_end_idx = 0
        else:
            history_end_idx = min(idx + 1, len(df))

        start_history_idx = max(0, history_end_idx - MAX_CHART_BARS)
        history = df.iloc[start_history_idx : history_end_idx].copy()

        self.chart.set(history, keep_drawings=not clear_drawings)

        with self._lock:
            self.full_df    = df
            self.replay_idx = history_end_idx
            self.elapsed    = 0.0

        if history_end_idx > 0:
            anchor_ts = history.index[-1]
            weekday_label = anchor_ts.strftime("%A")
            print(f"  → Jumped to Sim Time: {sim_time_naive.strftime('%Y-%m-%d %H:%M:%S')} {self.tz} (Visible bar: {anchor_ts.strftime('%H:%M')})")
        else:
            weekday_label = "—"

        self.chart.topbar["date_display"].set(weekday_label)
        self._update_timer(TF_SECS[self.tf])

        if was_playing:
            with self._lock:
                self.is_playing = True
            self.chart.topbar["btn_play"].set("⏸  Pause")

    # ─── BAR PUSH ─────────────────────────────────────────────────────────────
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

            # Advance the global clock precisely to the END of the bar we just pushed
            # This perfectly preserves multi-timeframe states (i.e. pushing 5m fast forwards the clock 5m)
            new_sim_naive = row.name + pd.Timedelta(seconds=TF_SECS[self.tf])
            new_sim_aware = new_sim_naive.tz_localize(TZ_MAP[self.tz])
            self.current_sim_time_utc = new_sim_aware.tz_convert("UTC")

            self.chart.topbar["date_display"].set(row.name.strftime("%A"))

        self.chart.update(bar)

    # ─── REPLAY LOOP ──────────────────────────────────────────────────────────
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

            wall_remaining = remaining_sim / max(speed, 1)
            self._update_timer(wall_remaining)
            time.sleep(TICK)

    # ─── CALLBACKS ────────────────────────────────────────────────────────────
    def _on_tf(self, chart):
        new_tf = self.chart.topbar["tf_switch"].value
        with self._lock:
            self.tf = new_tf
        self._load_for_sim_time(clear_drawings=False)

    def _on_tz(self, chart):
        new_tz = self.chart.topbar["tz_switch"].value
        with self._lock:
            old_tz = self.tz
            self.tz = new_tz
        print(f"Shifted Timezone from {old_tz} to {new_tz}. Adjusting X-Axis...")
        self._load_for_sim_time(clear_drawings=False)

    def _on_play(self, chart):
        with self._lock:
            if self.replay_idx >= len(self.full_df): return
            self.is_playing = not self.is_playing
            playing = self.is_playing
            if playing: self.elapsed = 0.0

        self.chart.topbar["btn_play"].set("⏸  Pause" if playing else "▶  Play")

    def _on_next(self, chart):
        with self._lock:
            if len(self.full_df) == 0: return
            self.elapsed = 0.0
        self._push_bar()
        with self._lock:
            tf_secs = TF_SECS[self.tf]
        self._update_timer(tf_secs)

    def _on_speed(self, chart):
        val = self.chart.topbar["speed_switch"].value
        with self._lock: self.speed = SPEED_MAP.get(val, 1)

    def _on_clear_drawings(self, chart):
        self.chart.toolbox.drawings.clear()
        self.chart.run_script(f"{self.chart.id}.toolBox?.clearDrawings();")
        print("All drawings cleared.")

    # ─── TIMER ────────────────────────────────────────────────────────────────
    def _update_timer(self, wall_seconds: float):
        s = max(0.0, wall_seconds)
        h, rem = divmod(int(s), 3600)
        m, sec  = divmod(rem, 60)
        label   = f"{h:02d}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"
        self.chart.watermark(f"Next bar: {label}", font_size=18, color="rgba(150,150,150,0.35)")

    # ─── RUN ──────────────────────────────────────────────────────────────────
    def run(self):
        self.chart.show(block=True)
        self._running = False


if __name__ == "__main__":
    app = ReplayApp()
    app.run()