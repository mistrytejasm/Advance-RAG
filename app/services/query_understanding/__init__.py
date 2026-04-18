"""query_understanding package — Phase 6.

Public API:
    from app.services.query_understanding import process_query
    from app.services.query_understanding.query_types import (
        QueryType, SearchRoute, QueryUnderstandingResult
    )
"""

from app.services.query_understanding.query_understanding import process_query

__all__ = ["process_query"]
