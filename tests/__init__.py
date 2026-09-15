"""Top-level hybrid-runtime test suite.

`agents/tests/` covers AgentCore in isolation and `edge/tests/` covers EMBER in
isolation. This package covers the seam between them under the conditions the
benchmark suite creates: payload integrity across the IPC boundary at volume,
and recovery from injected faults.
"""
