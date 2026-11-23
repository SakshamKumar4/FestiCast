"""
src/event_calendar.py  –  Event calendar engine for FestiCast
Supports: Indian Festivals, Global Holidays, Business Events, Custom Events
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

# ── Event data class ──────────────────────────────────────────────────────────
@dataclass
class Event:
    name: str
    category: str           # "festival" | "holiday" | "business" | "custom"
    base_date: date
    window_before: int = 7   # days before event
    window_after: int = 3    # days after event
    importance: float = 1.0  # 0-1 demand multiplier weight
    recurring: bool = True   # repeat every year?

    def dates_in_range(self, start: date, end: date) -> List[date]:
        """Return all event peak dates that fall within [start, end]."""
        results = []
        if self.recurring:
            for year in range(start.year, end.year + 1):
                try:
                    d = self.base_date.replace(year=year)
                except ValueError:
                    continue
                if start <= d <= end:
                    results.append(d)
        else:
            if start <= self.base_date <= end:
                results.append(self.base_date)
        return results


# ── Approximate lunar→Gregorian anchor dates (2013-2018) ─────────────────────
_DIWALI = {2013: date(2013, 11, 3), 2014: date(2014, 10, 23),
           2015: date(2015, 11, 11), 2016: date(2016, 10, 30),
           2017: date(2017, 10, 19), 2018: date(2018, 11, 7)}

_HOLI   = {2013: date(2013, 3, 27), 2014: date(2014, 3, 17),
           2015: date(2015, 3, 6),  2016: date(2016, 3, 24),
           2017: date(2017, 3, 13), 2018: date(2018, 3, 2)}

_NAVRATRI = {2013: date(2013, 10, 5), 2014: date(2014, 9, 25),
             2015: date(2015, 10, 13), 2016: date(2016, 10, 1),
             2017: date(2017, 9, 21), 2018: date(2018, 10, 10)}

_DUSSEHRA = {2013: date(2013, 10, 13), 2014: date(2014, 10, 3),
             2015: date(2015, 10, 22), 2016: date(2016, 10, 11),
             2017: date(2017, 9, 30),  2018: date(2018, 10, 19)}

_EID = {2013: date(2013, 8, 8), 2014: date(2014, 7, 28),
        2015: date(2015, 7, 18), 2016: date(2016, 7, 6),
        2017: date(2017, 6, 25), 2018: date(2018, 6, 15)}

_GANESH = {2013: date(2013, 9, 9), 2014: date(2014, 8, 29),
           2015: date(2015, 9, 17), 2016: date(2016, 9, 5),
           2017: date(2017, 8, 25), 2018: date(2018, 9, 13)}

_RAKSHA = {2013: date(2013, 8, 21), 2014: date(2014, 8, 10),
           2015: date(2015, 8, 29), 2016: date(2016, 8, 18),
           2017: date(2017, 8, 7),  2018: date(2018, 8, 26)}

_EASTER = {2013: date(2013, 3, 31), 2014: date(2014, 4, 20),
           2015: date(2015, 4, 5),  2016: date(2016, 3, 27),
           2017: date(2017, 4, 16), 2018: date(2018, 4, 1)}


class MultiYearEvent:
    """Event whose date varies by year (lunar/computus calendars)."""

    def __init__(self, name: str, category: str,
                 year_map: Dict[int, date],
                 window_before: int = 7, window_after: int = 3,
                 importance: float = 1.0):
        self.name = name
        self.category = category
        self.year_map = year_map
        self.window_before = window_before
        self.window_after = window_after
        self.importance = importance

    def dates_in_range(self, start: date, end: date) -> List[date]:
        results = []
        for year, d in self.year_map.items():
            if start <= d <= end:
                results.append(d)
        return results


# ── Event Calendar ─────────────────────────────────────────────────────────────
class EventCalendar:
    """
    Centralised event registry.
    Event-aware feature generation queries this class.
    New calendars can be registered via .add_event() without touching the pipeline.
    """

    def __init__(self):
        self._events: List = []
        self._register_defaults()

    # ------------------------------------------------------------------
    def _register_defaults(self):
        # Indian Festivals (lunar – multi-year)
        self._events += [
            MultiYearEvent("Diwali",           "festival", _DIWALI,  window_before=14, window_after=5, importance=1.0),
            MultiYearEvent("Holi",             "festival", _HOLI,   window_before=7,  window_after=2, importance=0.9),
            MultiYearEvent("Navratri",         "festival", _NAVRATRI,window_before=9,  window_after=2, importance=0.85),
            MultiYearEvent("Dussehra",         "festival", _DUSSEHRA,window_before=5,  window_after=2, importance=0.8),
            MultiYearEvent("Eid",              "festival", _EID,    window_before=7,  window_after=3, importance=0.85),
            MultiYearEvent("Ganesh Chaturthi", "festival", _GANESH, window_before=5,  window_after=2, importance=0.75),
            MultiYearEvent("Raksha Bandhan",   "festival", _RAKSHA, window_before=5,  window_after=1, importance=0.7),
        ]
        # Global Holidays (fixed date – recurring)
        self._events += [
            Event("Christmas",    "holiday", date(2000, 12, 25), window_before=14, window_after=3, importance=0.95),
            Event("New Year",     "holiday", date(2000, 1, 1),   window_before=5,  window_after=2, importance=0.8),
            Event("Halloween",    "holiday", date(2000, 10, 31), window_before=7,  window_after=1, importance=0.6),
            Event("Black Friday", "holiday", date(2000, 11, 29), window_before=3,  window_after=1, importance=0.9),
            Event("Cyber Monday", "holiday", date(2000, 12, 2),  window_before=1,  window_after=1, importance=0.85),
            MultiYearEvent("Easter", "holiday", _EASTER, window_before=7, window_after=2, importance=0.65),
        ]
        # Business Events
        self._events += [
            Event("Republic Day",         "business", date(2000, 1, 26),  window_before=2, window_after=1, importance=0.5),
            Event("Independence Day",     "business", date(2000, 8, 15),  window_before=2, window_after=1, importance=0.5),
            Event("Valentine's Day",      "business", date(2000, 2, 14),  window_before=5, window_after=1, importance=0.6),
            Event("Mother's Day",         "business", date(2000, 5, 12),  window_before=5, window_after=1, importance=0.55),
        ]

    # ------------------------------------------------------------------
    def add_event(self, event) -> None:
        """Register a new event (Event or MultiYearEvent)."""
        self._events.append(event)

    def remove_event(self, name: str) -> None:
        self._events = [e for e in self._events if e.name != name]

    def list_events(self) -> List[str]:
        return [e.name for e in self._events]

    # ------------------------------------------------------------------
    def build_event_dataframe(self, date_index: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Returns a DataFrame indexed by date with columns:
          is_event, event_name, event_category, event_importance,
          days_until_event, days_since_event, event_window, festival_score
        """
        start = date_index.min().date()
        end   = date_index.max().date()

        # Collect all peak dates
        rows = []
        for ev in self._events:
            for peak in ev.dates_in_range(start, end):
                rows.append({
                    "peak_date":        peak,
                    "event_name":       ev.name,
                    "event_category":   ev.category,
                    "event_importance": ev.importance,
                    "window_before":    ev.window_before,
                    "window_after":     ev.window_after,
                })

        df = pd.DataFrame(index=date_index)
        df["is_event"]          = 0
        df["event_name"]        = ""
        df["event_category"]    = "none"
        df["event_importance"]  = 0.0
        df["days_until_event"]  = 999
        df["days_since_event"]  = 999
        df["event_window"]      = 0
        df["festival_score"]    = 0.0

        for r in rows:
            peak   = pd.Timestamp(r["peak_date"])
            wb     = r["window_before"]
            wa     = r["window_after"]
            imp    = r["event_importance"]

            win_start = peak - pd.Timedelta(days=wb)
            win_end   = peak + pd.Timedelta(days=wa)

            # Mark window
            mask = (df.index >= win_start) & (df.index <= win_end)
            df.loc[mask, "event_window"]     = 1
            df.loc[mask, "event_importance"] = np.maximum(
                df.loc[mask, "event_importance"].values, imp)
            df.loc[mask, "event_name"]     = r["event_name"]
            df.loc[mask, "event_category"] = r["event_category"]

            # Peak date itself
            if peak in df.index:
                df.loc[peak, "is_event"] = 1

            # Days until / since
            for d in df.index:
                diff = (peak - d).days
                if 0 <= diff <= wb:                      # approaching
                    if diff < df.loc[d, "days_until_event"]:
                        df.loc[d, "days_until_event"] = diff
                elif -wa <= diff < 0:                    # just passed
                    since = -diff
                    if since < df.loc[d, "days_since_event"]:
                        df.loc[d, "days_since_event"] = since

        # Festival score: combo of importance + proximity
        in_window = df["event_window"] == 1
        df.loc[in_window, "festival_score"] = (
            df.loc[in_window, "event_importance"] *
            (1 / (1 + np.minimum(
                df.loc[in_window, "days_until_event"],
                df.loc[in_window, "days_since_event"]
            )))
        )

        return df


# ── Singleton accessor ────────────────────────────────────────────────────────
_calendar: Optional[EventCalendar] = None

def get_calendar() -> EventCalendar:
    global _calendar
    if _calendar is None:
        _calendar = EventCalendar()
    return _calendar
