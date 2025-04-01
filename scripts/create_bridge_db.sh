#!/bin/bash

DB_NAME="$1"

if [[ -z "$DB_NAME" ]]; then
  echo "Usage: $0 <db_name>"
  exit 1
fi

if [[ -f "$DB_NAME" ]]; then
  echo "Database '$DB_NAME' already exists. Aborting."
  exit 1
fi

sqlite3 "$DB_NAME" <<EOF
CREATE TABLE bridge_parameters (
    uuid TEXT PRIMARY KEY,
    deck_width TEXT,
    bottom_shift_percentage REAL,
    arch_fractions TEXT,
    pier_size INTEGER,
    circular_arch BOOLEAN,
    arch_height_fraction REAL
);
EOF
