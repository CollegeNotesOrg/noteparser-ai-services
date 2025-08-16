#!/bin/bash
set -e

# Create multiple databases for different services
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" <<-EOSQL
    CREATE DATABASE ragflow;
    CREATE DATABASE deepwiki;
    CREATE DATABASE dolphin;
    CREATE DATABASE langextract;
    
    -- Enable pgvector extension for vector similarity search
    \c ragflow;
    CREATE EXTENSION IF NOT EXISTS vector;
    
    \c deepwiki;
    CREATE EXTENSION IF NOT EXISTS pg_trgm;
    CREATE EXTENSION IF NOT EXISTS fuzzystrmatch;
    
    -- Grant permissions
    GRANT ALL PRIVILEGES ON DATABASE ragflow TO noteparser;
    GRANT ALL PRIVILEGES ON DATABASE deepwiki TO noteparser;
    GRANT ALL PRIVILEGES ON DATABASE dolphin TO noteparser;
    GRANT ALL PRIVILEGES ON DATABASE langextract TO noteparser;
EOSQL

echo "Databases created successfully"