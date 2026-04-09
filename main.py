# Main application entry point
# main.py
import argparse
import asyncio
from dotenv import load_dotenv

load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="SYRA Adaptive AI Tutor v2")
    parser.add_argument("--mode",    default="live",
                        choices=["live", "onboard", "report"])
    parser.add_argument("--subject", default="Mathematics")
    parser.add_argument("--grade",   type=int, default=10)
    parser.add_argument("--student", default="student_001")
    args = parser.parse_args()

    if args.mode == "live":
        asyncio.run(_run_live(args.subject, args.grade, args.student))
    elif args.mode == "onboard":
        _run_onboard(args.subject, args.student)
    elif args.mode == "report":
        _run_report(args.student)


async def _run_live(subject: str, grade: int, student_id: str):
    from onboarding.questionnaire  import run_onboarding
    from intake.session_checker    import run_session_check
    from memory.profile_manager    import ProfileManager
    from memory.session_memory     import SessionMemory
    import memory.belief_graph as bg_module
    from voice.live_session        import FullDuplexSession

    pm      = ProfileManager(student_id)
    profile, belief_graph = run_onboarding(student_id, subject)
    pm.profile = profile

    archetype = profile.get("ipc", {}).get("archetype", "lina")
    name      = student_id.replace("_", " ").title()

    # Per-session state check
    session_ctx = run_session_check(student_name=name, archetype=archetype)

    sm   = SessionMemory(profile, subject)
    live = FullDuplexSession(
        profile, pm, sm, belief_graph,
        session_ctx, subject, grade
    )

    try:
        await live.run()
    except KeyboardInterrupt:
        print("\n  Interrupted — saving...")
        await live._end_session()


def _run_onboard(subject: str, student_id: str):
    from onboarding.questionnaire import run_onboarding
    profile, belief_graph = run_onboarding(student_id, subject)
    arch = profile.get("ipc", {}).get("archetype", "unknown")
    print(f"\n  Done. Archetype: {arch}")
    concepts = list(belief_graph.get("concepts", {}).keys())
    print(f"  Belief graph seeded: {concepts}")


def _run_report(student_id: str):
    from feedback.report_generator import generate_report
    generate_report(student_id)


if __name__ == "__main__":
    main()