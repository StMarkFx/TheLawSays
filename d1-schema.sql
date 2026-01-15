
        CREATE TABLE IF NOT EXISTS chunks (
            id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            jurisdiction TEXT NOT NULL,
            text TEXT NOT NULL,
            metadata TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_chunks_jurisdiction ON chunks(jurisdiction);
        CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source);
        