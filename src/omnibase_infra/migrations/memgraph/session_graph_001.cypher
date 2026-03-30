// Session graph schema migration 001
// Part of the Multi-Session Coordination Layer (OMN-6850, Task 8)
//
// Node labels: Session, Task, File, PullRequest, Repository
// Relationships: WORKS_ON, TOUCHES, DEPENDS_ON, PRODUCED, BELONGS_TO

// --- Uniqueness constraints ---
CREATE CONSTRAINT ON (s:Session) ASSERT s.session_id IS UNIQUE;
CREATE CONSTRAINT ON (t:Task) ASSERT t.task_id IS UNIQUE;
CREATE CONSTRAINT ON (f:File) ASSERT f.path IS UNIQUE;
CREATE CONSTRAINT ON (p:PullRequest) ASSERT p.pr_id IS UNIQUE;
CREATE CONSTRAINT ON (r:Repository) ASSERT r.name IS UNIQUE;

// --- Performance indexes ---
CREATE INDEX ON :Task(status);
CREATE INDEX ON :Session(last_activity);
