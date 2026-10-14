import asyncio
import json
import sys
import time
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def decoded(result):
    assert not result.isError, result.content
    value = result.structuredContent
    return value if value is not None else json.loads(result.content[0].text)


def test_real_stdio_protocol_background_worker_and_images(sample, tmp_path):
    store, job_id, image, mesh = sample
    config = tmp_path / "profile.json"
    config.write_text(json.dumps(store.load(job_id)["config"]))
    params = StdioServerParameters(
        env={"PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")},
        command=sys.executable,
        args=[
            "-m",
            "agile3d.mcp_server",
            "--root",
            str(store.root),
            "--config",
            str(config),
            "--input-root",
            str(tmp_path),
        ],
    )

    async def exercise():
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = {t.name for t in (await session.list_tools()).tools}
                assert {
                    "inspect_candidate",
                    "review_candidate",
                    "start_hoi_run_stage",
                } <= tools
                denied = await session.call_tool(
                    "create_agile_job",
                    {"references": ["/etc/passwd"], "description": "object"},
                )
                assert denied.isError
                started = decoded(
                    await session.call_tool(
                        "start_candidate",
                        {
                            "job_id": job_id,
                            "image_path": str(image),
                            "mesh_path": str(mesh),
                        },
                    )
                )
                assert started["status"] == "queued"
                deadline = time.monotonic() + 30
                while time.monotonic() < deadline:
                    job = decoded(
                        await session.call_tool("get_job", {"job_id": job_id})
                    )
                    if job["status"] not in ("queued", "running"):
                        break
                    await asyncio.sleep(0.1)
                assert job["status"] == "awaiting_review", job
                result = await session.call_tool(
                    "inspect_candidate", {"job_id": job_id}
                )
                assert not result.isError
                assert sum(c.type == "image" for c in result.content) == 3
                evidence = json.loads(
                    next(c.text for c in result.content if c.type == "text")
                )
                review = dict(
                    evidence_digest=evidence["evidence_digest"],
                    decision="accept",
                    score=8.0,
                    matches_reference=True,
                    no_hand_geometry=True,
                    consistent_views=True,
                    feedback="All observed surfaces are consistent.",
                )
                accepted = decoded(
                    await session.call_tool(
                        "review_candidate", {"job_id": job_id, "review": review}
                    )
                )
                assert accepted["status"] == "accepted"
                exported = decoded(
                    await session.call_tool("export_accepted_asset", {"job_id": job_id})
                )
                assert (Path(exported["directory"]) / "textured_mesh.obj").exists()
                final = decoded(await session.call_tool("finalize_best_asset", {"job_id": job_id}))
                assert final["status"] == "delivered" and final["quality_status"] == "accepted"
                assert (Path(final["directory"]) / "textured_mesh.glb").exists()

    asyncio.run(exercise())
