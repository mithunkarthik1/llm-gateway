-- =====================================================================
-- Enterprise ERP Knowledge Base PostgreSQL Migration Script
-- Primary Source of Truth for Relational Schemas and Hybrid Segments
-- =====================================================================

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ---------------------------------------------------------------------
-- A. domains: Domain-level enterprise master metadata
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS domains (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    domain_id VARCHAR(64) NOT NULL UNIQUE,
    system_name VARCHAR(128) NOT NULL DEFAULT 'Enterprise ERP',
    domain_name VARCHAR(128) NOT NULL,
    description TEXT,
    keywords TEXT[] DEFAULT '{}',
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_domains_domain_id ON domains(domain_id);
CREATE INDEX IF NOT EXISTS idx_domains_is_active ON domains(is_active);

-- ---------------------------------------------------------------------
-- B. knowledge_tables: Table-level schema metadata & definitions
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS knowledge_tables (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    domain_id UUID NOT NULL REFERENCES domains(id) ON DELETE CASCADE,
    table_name VARCHAR(128) NOT NULL,
    description TEXT,
    keywords TEXT[] DEFAULT '{}',
    sql_definition TEXT NOT NULL,
    content_hash VARCHAR(64) NOT NULL,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_knowledge_tables_domain_table UNIQUE (domain_id, table_name)
);

CREATE INDEX IF NOT EXISTS idx_knowledge_tables_table_name ON knowledge_tables(table_name);
CREATE INDEX IF NOT EXISTS idx_knowledge_tables_domain_id ON knowledge_tables(domain_id);
CREATE INDEX IF NOT EXISTS idx_knowledge_tables_is_active ON knowledge_tables(is_active);

-- ---------------------------------------------------------------------
-- C. knowledge_columns: Column metadata with data types & constraints
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS knowledge_columns (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    table_id UUID NOT NULL REFERENCES knowledge_tables(id) ON DELETE CASCADE,
    column_name VARCHAR(128) NOT NULL,
    data_type VARCHAR(128) NOT NULL,
    description TEXT,
    is_primary_key BOOLEAN NOT NULL DEFAULT FALSE,
    is_foreign_key BOOLEAN NOT NULL DEFAULT FALSE,
    references_table VARCHAR(128),
    references_column VARCHAR(128),
    keywords TEXT[] DEFAULT '{}',
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_knowledge_columns_table_col UNIQUE (table_id, column_name)
);

CREATE INDEX IF NOT EXISTS idx_knowledge_columns_table_id ON knowledge_columns(table_id);
CREATE INDEX IF NOT EXISTS idx_knowledge_columns_column_name ON knowledge_columns(column_name);
CREATE INDEX IF NOT EXISTS idx_knowledge_columns_is_pk ON knowledge_columns(is_primary_key);
CREATE INDEX IF NOT EXISTS idx_knowledge_columns_is_fk ON knowledge_columns(is_foreign_key);

-- ---------------------------------------------------------------------
-- D. table_dependencies: Table relationships & referential hierarchy
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS table_dependencies (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_table_id UUID NOT NULL REFERENCES knowledge_tables(id) ON DELETE CASCADE,
    referenced_table_id UUID NOT NULL REFERENCES knowledge_tables(id) ON DELETE CASCADE,
    relationship_type VARCHAR(64) NOT NULL DEFAULT 'FOREIGN_KEY',
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_table_dependencies UNIQUE (source_table_id, referenced_table_id, relationship_type)
);

CREATE INDEX IF NOT EXISTS idx_table_dependencies_source ON table_dependencies(source_table_id);
CREATE INDEX IF NOT EXISTS idx_table_dependencies_ref ON table_dependencies(referenced_table_id);

-- ---------------------------------------------------------------------
-- E. knowledge_segments: Hybrid chunking hierarchy (Section -> Topic -> Segment -> UUID)
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS knowledge_segments (
    id UUID PRIMARY KEY,
    domain_id UUID NOT NULL REFERENCES domains(id) ON DELETE CASCADE,
    table_id UUID REFERENCES knowledge_tables(id) ON DELETE SET NULL,
    section VARCHAR(128) NOT NULL,
    topic VARCHAR(128) NOT NULL,
    segment_type VARCHAR(64) NOT NULL,
    segment_name VARCHAR(256) NOT NULL,
    content TEXT NOT NULL,
    content_hash VARCHAR(64) NOT NULL,
    version INTEGER NOT NULL DEFAULT 1,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_knowledge_segments_domain ON knowledge_segments(domain_id);
CREATE INDEX IF NOT EXISTS idx_knowledge_segments_table ON knowledge_segments(table_id);
CREATE INDEX IF NOT EXISTS idx_knowledge_segments_section ON knowledge_segments(section);
CREATE INDEX IF NOT EXISTS idx_knowledge_segments_topic ON knowledge_segments(topic);
CREATE INDEX IF NOT EXISTS idx_knowledge_segments_type ON knowledge_segments(segment_type);
CREATE INDEX IF NOT EXISTS idx_knowledge_segments_hash ON knowledge_segments(content_hash);
CREATE INDEX IF NOT EXISTS idx_knowledge_segments_is_active ON knowledge_segments(is_active);

-- ---------------------------------------------------------------------
-- F. segment_versions: Historical segment change tracking
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS segment_versions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    segment_id UUID NOT NULL REFERENCES knowledge_segments(id) ON DELETE CASCADE,
    version INTEGER NOT NULL,
    content TEXT NOT NULL,
    content_hash VARCHAR(64) NOT NULL,
    changed_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_segment_versions_seg_ver UNIQUE (segment_id, version)
);

CREATE INDEX IF NOT EXISTS idx_segment_versions_segment_id ON segment_versions(segment_id);

-- ---------------------------------------------------------------------
-- G. embedding_jobs: Embedding & FAISS synchronization queue
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS embedding_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    segment_id UUID NOT NULL REFERENCES knowledge_segments(id) ON DELETE CASCADE,
    content_hash VARCHAR(64) NOT NULL,
    operation VARCHAR(32) NOT NULL CHECK (operation IN ('INSERT', 'UPDATE', 'DELETE')),
    status VARCHAR(32) NOT NULL DEFAULT 'PENDING' CHECK (status IN ('PENDING', 'PROCESSING', 'COMPLETED', 'FAILED')),
    retry_count INTEGER NOT NULL DEFAULT 0,
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    processed_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX IF NOT EXISTS idx_embedding_jobs_status ON embedding_jobs(status);
CREATE INDEX IF NOT EXISTS idx_embedding_jobs_segment_id ON embedding_jobs(segment_id);

-- ---------------------------------------------------------------------
-- H. segment_vectors: UUID-to-FAISS integer ID mapping
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS segment_vectors (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    segment_id UUID NOT NULL REFERENCES knowledge_segments(id) ON DELETE CASCADE,
    embedding_model VARCHAR(128) NOT NULL,
    model_version VARCHAR(64) NOT NULL DEFAULT 'v1',
    faiss_vector_id BIGINT NOT NULL,
    content_hash VARCHAR(64) NOT NULL,
    indexed_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_segment_vectors_model_seg UNIQUE (embedding_model, segment_id),
    CONSTRAINT uq_segment_vectors_faiss_id UNIQUE (embedding_model, faiss_vector_id)
);

CREATE INDEX IF NOT EXISTS idx_segment_vectors_segment_id ON segment_vectors(segment_id);
CREATE INDEX IF NOT EXISTS idx_segment_vectors_faiss_id ON segment_vectors(faiss_vector_id);
