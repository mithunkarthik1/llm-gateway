import hashlib
import uuid
from typing import Any, Dict, List, Optional

# Stable namespace for deterministic segment UUIDs across content edits
SEGMENT_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")


def compute_sha256(text: str) -> str:
    """Computes normalized SHA-256 hash of text content."""
    clean = text.strip().encode("utf-8")
    return hashlib.sha256(clean).hexdigest()


def generate_stable_segment_uuid(domain_id: str, topic: str, segment_name: str) -> uuid.UUID:
    """
    Generates a deterministic, collision-free UUIDv5 for a logical segment identity.
    Ensures that a segment retains its exact UUID across multiple content updates.
    """
    identity_key = f"{domain_id.lower().strip()}:{topic.lower().strip()}:{segment_name.lower().strip()}"
    return uuid.uuid5(SEGMENT_NAMESPACE, identity_key)


class KnowledgeChunk:
    """Represents a structured hybrid chunk ready for DB persistence & vectorization."""

    def __init__(
        self,
        segment_uuid: uuid.UUID,
        domain_id: str,
        section: str,
        topic: str,
        segment_type: str,
        segment_name: str,
        content: str,
        table_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.segment_uuid = segment_uuid
        self.domain_id = domain_id
        self.section = section
        self.topic = topic
        self.segment_type = segment_type
        self.segment_name = segment_name
        self.content = content.strip()
        self.content_hash = compute_sha256(self.content)
        self.table_name = table_name
        self.metadata = metadata or {}


def build_domain_chunk(domain_data: Dict[str, Any]) -> KnowledgeChunk:
    """Builds a domain-level hybrid segment."""
    dom_id = domain_data.get("domain_id", "").strip()
    dom_name = domain_data.get("domain_name", dom_id).strip()
    system = domain_data.get("system", "Enterprise ERP").strip()
    desc = domain_data.get("description", "").strip()
    keywords = domain_data.get("keywords", [])
    tbls = domain_data.get("table_names", [])

    section = dom_name
    topic = f"{dom_name} Overview"
    seg_name = f"{dom_id}.overview"
    seg_uuid = generate_stable_segment_uuid(dom_id, topic, seg_name)

    content = (
        f"Section: {section}\n"
        f"Topic: {topic}\n"
        f"Segment: {seg_name}\n\n"
        f"System: {system}\n"
        f"Domain ID: {dom_id}\n"
        f"Domain Name: {dom_name}\n"
        f"Description: {desc}\n"
        f"Master Tables: {', '.join(tbls)}\n"
        f"Keywords: {', '.join(keywords)}"
    )

    return KnowledgeChunk(
        segment_uuid=seg_uuid,
        domain_id=dom_id,
        section=section,
        topic=topic,
        segment_type="domain",
        segment_name=seg_name,
        content=content,
        metadata={
            "domain_id": dom_id,
            "domain_name": dom_name,
            "segment_type": "domain",
            "identity_key": f"{dom_id}:domain:overview",
            "table_count": len(tbls)
        }
    )


def build_table_chunk(domain_data: Dict[str, Any], table_data: Dict[str, Any]) -> KnowledgeChunk:
    """Builds a table-level hybrid segment."""
    dom_id = domain_data.get("domain_id", "").strip()
    dom_name = domain_data.get("domain_name", dom_id).strip()
    system = domain_data.get("system", "Enterprise ERP").strip()

    tname = table_data.get("table_name", "").strip()
    desc = table_data.get("description", "").strip()
    keywords = table_data.get("keywords", [])
    deps = table_data.get("dependencies", [])
    cols = table_data.get("columns", [])
    raw_sql = table_data.get("sql", "").strip()

    section = dom_name
    topic = tname.replace("_", " ").title()
    seg_name = f"{tname}.table_definition"
    seg_uuid = generate_stable_segment_uuid(dom_id, topic, seg_name)

    content = (
        f"Section: {section}\n"
        f"Topic: {topic}\n"
        f"Segment: {seg_name}\n\n"
        f"System: {system}\n"
        f"Domain: {dom_name} ({dom_id})\n"
        f"Table: {tname}\n"
        f"Description: {desc}\n"
        f"Foreign Key Dependencies: {', '.join(deps) if deps else 'None'}\n"
        f"Columns: {', '.join(cols)}\n"
        f"Keywords: {', '.join(keywords)}\n\n"
        f"SQL Definition:\n{raw_sql}"
    )

    return KnowledgeChunk(
        segment_uuid=seg_uuid,
        domain_id=dom_id,
        section=section,
        topic=topic,
        segment_type="table",
        segment_name=seg_name,
        content=content,
        table_name=tname,
        metadata={
            "domain_id": dom_id,
            "domain_name": dom_name,
            "table_name": tname,
            "segment_type": "table",
            "identity_key": f"{dom_id}:{tname}:definition",
            "dependencies": deps
        }
    )


def build_column_chunk(domain_data: Dict[str, Any], column_data: Dict[str, Any]) -> KnowledgeChunk:
    """Builds a granular column-level hybrid segment."""
    dom_id = domain_data.get("domain_id", "").strip()
    dom_name = domain_data.get("domain_name", dom_id).strip()

    tname = column_data.get("table_name", "").strip()
    cname = column_data.get("column_name", "").strip()
    dtype = column_data.get("data_type", "").strip()
    cdesc = column_data.get("description", "").strip()
    ckw = column_data.get("keywords", [])
    is_pk = column_data.get("is_primary_key", False)
    is_fk = column_data.get("is_foreign_key", False)
    ref = column_data.get("references")

    section = dom_name
    topic = tname.replace("_", " ").title()
    seg_name = f"{tname}.{cname}"
    seg_uuid = generate_stable_segment_uuid(dom_id, topic, seg_name)

    pk_str = "Yes" if is_pk else "No"
    fk_str = f"Yes (References: {ref})" if is_fk else "No"

    content = (
        f"Section: {section}\n"
        f"Topic: {topic}\n"
        f"Segment: {seg_name}\n\n"
        f"Table: {tname}\n"
        f"Column: {cname}\n"
        f"Data Type: {dtype}\n"
        f"Primary Key: {pk_str}\n"
        f"Foreign Key: {fk_str}\n"
        f"Description: {cdesc}\n"
        f"Keywords: {', '.join(ckw) if ckw else cname}"
    )

    return KnowledgeChunk(
        segment_uuid=seg_uuid,
        domain_id=dom_id,
        section=section,
        topic=topic,
        segment_type="column",
        segment_name=seg_name,
        content=content,
        table_name=tname,
        metadata={
            "domain_id": dom_id,
            "domain_name": dom_name,
            "table_name": tname,
            "column_name": cname,
            "data_type": dtype,
            "is_primary_key": is_pk,
            "is_foreign_key": is_fk,
            "segment_type": "column",
            "identity_key": f"{dom_id}:{tname}:{cname}"
        }
    )


def build_relationship_chunk(
    domain_data: Dict[str, Any],
    source_table: str,
    target_table: str,
    fk_col: str,
    rel_type: str = "FOREIGN_KEY"
) -> KnowledgeChunk:
    """Builds a relational foreign-key dependency hybrid segment."""
    dom_id = domain_data.get("domain_id", "").strip()
    dom_name = domain_data.get("domain_name", dom_id).strip()

    section = dom_name
    topic = source_table.replace("_", " ").title()
    seg_name = f"{source_table}.ref_{target_table}"
    seg_uuid = generate_stable_segment_uuid(dom_id, topic, seg_name)

    content = (
        f"Section: {section}\n"
        f"Topic: {topic}\n"
        f"Segment: {seg_name}\n\n"
        f"Source Table: {source_table}\n"
        f"Foreign Key Column: {fk_col}\n"
        f"Referenced Parent Table: {target_table}\n"
        f"Relationship Type: {rel_type}\n"
        f"Relational Rule: Table '{source_table}' directly depends on parent table '{target_table}' via key '{fk_col}'."
    )

    return KnowledgeChunk(
        segment_uuid=seg_uuid,
        domain_id=dom_id,
        section=section,
        topic=topic,
        segment_type="relationship",
        segment_name=seg_name,
        content=content,
        table_name=source_table,
        metadata={
            "domain_id": dom_id,
            "domain_name": dom_name,
            "source_table": source_table,
            "referenced_table": target_table,
            "fk_column": fk_col,
            "segment_type": "relationship",
            "identity_key": f"{dom_id}:{source_table}:ref_{target_table}"
        }
    )
