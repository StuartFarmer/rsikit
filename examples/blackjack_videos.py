"""Compatibility entry point for saved generation videos."""

from research.elitesearch.videos import (
    evaluation_panel as evaluation_panel,
)
from research.elitesearch.videos import (
    export as export,
)
from research.elitesearch.videos import (
    main as main,
)
from research.elitesearch.videos import (
    render_policy as render_policy,
)
from research.elitesearch.videos import (
    trace_environment as trace_environment,
)
from research.elitesearch.videos import (
    write_index as write_index,
)

if __name__ == "__main__":
    main()
