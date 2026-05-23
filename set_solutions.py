"""Set the kiosk's solutions catalog (the things ECHO can match a
visitor's question against).

REPLACES the solutions table with exactly the list below. Run on the Pi:

    python set_solutions.py

This catalog is NOT shown on the idle dashboard (that's the projects
list -- see set_projects.py). It's searched only when a visitor asks
about a solution in chat, e.g. "do you have something for container
security?" -> ECHO names the matching solution.

Each entry is (name, description, domain, link). Edit and re-run to
change the catalog. NOTE: the links were transcribed from a screenshot
and should be double-checked before relying on them.
"""

from __future__ import annotations

from database import FaceDB


SOLUTIONS = [
    ("Cloud Shield Security Platform",
     "An AI-powered cloud security platform that automatically detects "
     "credential exposures, backup gaps, and misconfigurations across AWS, "
     "then routes each finding through a 12-node LangGraph pipeline and an "
     "ML confidence gate to either auto-remediate via Harness CI/CD or queue "
     "for human review, closing the loop with a full ServiceNow INC + CHG "
     "audit trail in under 6 minutes.",
     "Operations", "https://cloudshield.aleksops.com"),

    ("AI Based DB Performance Management",
     "AI-powered database performance triage platform that ingests AWR / "
     "slow-query diagnostics and auto-generates ranked runbooks with "
     "remediation steps across Oracle, MySQL, PostgreSQL, MariaDB, and SQL "
     "Server.",
     "Operations", "https://dbperf.aleksops.com"),

    ("AWS Transform Hub",
     "Cloud migration acceleration platform with Discovery, Governance, "
     "Testing, Day-2 Ops, and Agent Studio modules that tracks and "
     "orchestrates AWS migration jobs submitted by the Cobalt engine.",
     "Migration", "https://transform.aleksops.com"),

    ("Cobalt Migration Accelerator",
     "End-to-end AI-driven cloud migration platform that discovers "
     "on-premises workloads, generates migration plans, and executes "
     "containerization and database replatforming to EKS and Aurora via a "
     "42-step agentic pipeline.",
     "Migration", "https://migration.aleksops.com"),

    ("Container Vulnerability",
     "ContainerVul is an AI-powered container security platform that scans "
     "Docker images for CVEs, triages vulnerabilities by severity, "
     "auto-creates ServiceNow incidents, and generates automated Dockerfile "
     "fix pull requests on GitHub.",
     "Operations", "https://vulcon.aleksops.com"),

    ("Well Architect Framework",
     "AI-powered AWS Well-Architected Framework advisor that automates "
     "multi-account cloud assessments, generates remediation "
     "recommendations, and delivers enterprise-grade compliance and cost "
     "optimization insights.",
     "Well Architect", "https://wellarchitect.online"),

    ("DB Migration",
     "AI-powered database migration analysis and planning tool with schema "
     "comparison and migration path generation.",
     "Transformation", "https://dbmigration.aleksops.com"),

    ("Teradata to BigQuery",
     "Migration tooling for moving Teradata data warehouses to Google "
     "BigQuery.",
     "Migration", ""),

    ("EKS Operations",
     "Kubernetes AIOps platform for AI-driven EKS and AKS cluster "
     "management, vulnerability scanning, and operational automation.",
     "Operations", "https://aleksops.com"),

    ("Windows Vulnerability Management",
     "Enterprise Windows Server vulnerability scanner and remediation tool "
     "with Claude AI and multi-account AWS support.",
     "Operations", "https://winvul.streamlit.app"),

    ("Multi-Cloud Operation Platform",
     "Enterprise AI ops agent with an 8-agent team, ServiceNow integration, "
     "SLO tracking, and AWS incident automation.",
     "Operations", ""),

    ("AI Agents Based Application Development Platform",
     "AI-powered application development platform that orchestrates 14 "
     "specialized agents (Researcher, PM, Developer, Security, QA, "
     "Compliance, DevOps, etc.) to take a feature request and deliver "
     "production-ready code with full quality gates. A \"Feature Factory\": "
     "you describe what to build and the 14-agent pipeline handles research, "
     "PM scoping, coding, security review, testing, compliance (PCI-DSS / "
     "GDPR), DBA migration scripts, DevOps manifests, and live preview, "
     "autonomously across up to 3 iterative cycles. Multi-model: Claude Opus "
     "4.7 for coding, GPT-4o for research, local Llama for lighter tasks.",
     "Development", "https://agents.aleksops.com"),

    ("FinOps and Compliance",
     "A unified AWS governance platform that simultaneously manages FinOps "
     "cost optimization, multi-account compliance enforcement, and "
     "AI-powered security threat remediation from a single pane of glass, "
     "replacing the fragmented toolset enterprises use across Security Hub, "
     "Cost Explorer, Config, and Control Tower. Built for enterprises "
     "running hundreds of AWS accounts where compliance drift, overspend, "
     "and delayed threat response are persistent risks.",
     "FinOps", "https://finopsncompliance.streamlit.app"),

    ("Application Lifecycle Tracker",
     "An AI-powered enterprise software EOL/EOS risk management platform "
     "that tracks end-of-life and end-of-support dates for 321+ software "
     "components (Windows, Linux, Oracle, SQL Server, PostgreSQL, Tomcat, "
     ".NET, Node.js, and more), scores each version 0-100 by risk proximity, "
     "and generates AI-driven upgrade/migration recommendations, helping "
     "enterprises stay compliant with PCI-DSS, HIPAA, NIST, and ISO 27001 "
     "before vendor support windows close.",
     "Migration", "https://eva.aleksops.com"),

    ("AI Estimator for LLMs",
     "An enterprise Gen AI investment calculator that computes the true "
     "Total Cost of Ownership and ROI for AI deployments across 8 dimensions "
     "(API costs, infrastructure, development, data management, operations, "
     "and organizational overhead). It prevents enterprises from "
     "underestimating AI costs and overestimating benefits by delivering "
     "data-driven financial analysis that CFOs and CIOs can take to a board.",
     "Planning", "https://aiestimators.com"),

    ("AI Contract Lifecycle Management",
     "An Infosys Cobalt-branded AI platform that automates the full contract "
     "lifecycle (extraction, drafting, risk scoring, and comparison) using "
     "four named AI agents, replacing the manual legal review bottleneck "
     "that slows deal closure and lets risky terms slip through. It "
     "centralizes contract intelligence so legal, procurement, and "
     "compliance teams work from a single repository with expiration "
     "tracking, clause reuse, and audit trails.",
     "Contract Management", "https://contractmgmt.streamlit.app/"),
]


def main():
    db = FaceDB()
    try:
        existing = db.list_solutions()
        for row in existing:
            db.delete_solution(row[0])
        for i, (name, desc, domain, link) in enumerate(SOLUTIONS, start=1):
            db.add_solution(name, desc, domain, link, ordering=i)
        print(f"solutions: removed {len(existing)}, inserted {len(SOLUTIONS)}")
        for i, (name, *_rest) in enumerate(SOLUTIONS, start=1):
            print(f"  {i:>2}. {name}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
