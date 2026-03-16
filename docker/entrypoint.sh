#!/bin/bash
set -e

echo "=== QTS Entrypoint ==="

# Ensure database tables exist using SQLAlchemy create_all (async engine).
# This is safe to call repeatedly — create_all uses checkfirst=True by default.
if [ -n "$DATABASE_URL" ]; then
    echo "Ensuring database tables..."
    python -c "
import asyncio
from qts.db.engine import ensure_tables
asyncio.run(ensure_tables())
print('Database tables ready.')
" || echo "WARNING: Database table creation failed — continuing anyway"
fi

# Execute the main command
exec "$@"
