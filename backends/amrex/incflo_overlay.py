"""Generate a narrowly modified incflo source tree from an immutable upstream.

The source archive remains untouched. Every replacement has an exact count;
upstream drift is a configuration error, never a best-effort patch.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def replace(text: str, old: str, new: str, count: int = 1) -> str:
    if text.count(old) != count:
        raise ValueError(f"upstream drift: expected {count} copies of {old!r}")
    return text.replace(old, new)


def generate(source: Path, destination: Path) -> None:
    if source.resolve() == destination.resolve():
        raise ValueError("the upstream source must remain untouched")
    generated = {
        path.relative_to(source): path.read_bytes()
        for path in source.rglob("*")
        if path.is_file()
    }

    def edit(name: str, changes: list[tuple]) -> None:
        original = (source / name).read_text()
        for change in changes:
            original = replace(original, *change)
        generated[Path(name)] = original.encode()

    edit(
        "incflo.H",
        [
            (
                "bool include_pressure_gradient = true);",
                (
                    "bool include_pressure_gradient = true,\n"
                    "                                       amrex::Real force_time = -1,\n"
                    "                                       amrex::Real force_time_end = -1);"
                ),
                2,
            ),
        ],
    )
    edit(
        "incflo_compute_forces.cpp",
        [
            ("#include <incflo.H>", "#include <incflo.H>\n#include <ns_case.H>"),
            (
                "bool include_pressure_gradient)\n",
                (
                    "bool include_pressure_gradient,\n"
                    "                                 Real force_time, Real force_time_end)\n"
                ),
                2,
            ),
            (
                "*tracer_old[lev], *tracer_new[lev], include_pressure_gradient);",
                "*tracer_old[lev], *tracer_new[lev], include_pressure_gradient, force_time, force_time_end);",
            ),
            (
                "    GpuArray<Real,3> l_gravity",
                (
                    "    if (force_time < 0) force_time = m_cur_time;\n"
                    "    GpuArray<Real,3> l_gravity"
                ),
            ),
            (
                "                });\n            }\n    }\n}\n",
                (
                    "                });\n            }\n    }\n"
                    "    ns_case::add_force(lev, vel_forces, geom[lev], force_time, force_time_end, m_mu);\n}\n"
                ),
            ),
        ],
    )
    # Godunov traces with the old-time force, then updates with a midpoint force.
    # MOL predicts using f(t_n), and its corrector uses [f(t_n)+f(t_n+dt)]/2.
    # Pressure is NOT endpoint-averaged; the upstream pressure update is retained.
    edit(
        "incflo_update_velocity.cpp",
        [
            (
                "get_density_nph_const(), get_tracer_old_const(), get_tracer_new_const());",
                (
                    "get_density_nph_const(), get_tracer_old_const(), get_tracer_new_const(),\n"
                    '                           true, m_advection_type == "MOL" ? m_cur_time : m_cur_time + 0.5*m_dt);'
                ),
                2,
            ),
        ],
    )
    # The second occurrence is the corrector, always MOL.
    path = Path("incflo_update_velocity.cpp")
    text = generated[path].decode()
    marker = 'true, m_advection_type == "MOL" ? m_cur_time : m_cur_time + 0.5*m_dt);'
    first, second, tail = text.split(marker)
    generated[path] = (
        first + marker + second + "true, m_cur_time, new_time);" + tail
    ).encode()
    edit(
        "incflo_apply_corrector.cpp",
        [
            ("include_pressure_gradient);", "include_pressure_gradient, new_time);"),
        ],
    )
    edit(
        "utilities/io.cpp",
        [
            (
                'm_leveldata[lev]->tracer);\n            }\n            AMREX_D_TERM(pltscaVarsName.push_back("forcing_x");',
                (
                    "m_leveldata[lev]->tracer, false, m_cur_time);\n            }\n"
                    '            AMREX_D_TERM(pltscaVarsName.push_back("forcing_x");'
                ),
            ),
        ],
    )
    edit(
        "incflo_compute_dt.cpp",
        [
            ("#include <incflo.H>", "#include <incflo.H>\n#include <ns_case.H>"),
            (
                "if(! initialization && comb_cfl <= eps)",
                "if(! initialization && comb_cfl <= eps && ns_case::options().max_dt >= 1e99)",
            ),
            (
                "if (dt_new < eps)",
                "if (dt_new < eps && ns_case::options().max_dt >= 1e99 && ns_case::options().plot_times.empty())",
            ),
            (
                "    // Don't overshoot specified plot times",
                (
                    "    dt_new = ns_case::limit_dt(m_cur_time, dt_new, comb_cfl == 0);\n"
                    "    dt_new = ns_case::limit_plot_dt(m_cur_time, dt_new);\n\n"
                    "    // Don't overshoot specified plot times"
                ),
            ),
            (
                "dt_new = std::trunc((m_cur_time + dt_new) / m_plot_per_exact) * m_plot_per_exact - m_cur_time;",
                # Stop at the first event. Match writeNow's 1e-12 acceptance
                # tolerance: a frame already accepted a few ulps below its
                # nominal time must not cause a duplicate ~1e-15 step. min
                # also prevents this tolerance from enlarging a CFL bound.
                "dt_new = amrex::min(dt_new, (std::floor((m_cur_time + 1.e-12) / m_plot_per_exact) + 1) * m_plot_per_exact - m_cur_time);",
            ),
        ],
    )
    edit(
        "incflo_tagging.cpp",
        [
            ("#include <incflo.H>", "#include <incflo.H>\n#include <ns_case.H>"),
            (
                "    const auto   tagval",
                (
                    "    ns_case::tag_fixed_region(levc, tags, geom[levc]);\n\n"
                    "    const auto   tagval"
                ),
            ),
        ],
    )
    call = "ns_case::diagnose(get_velocity_new_const(), geom, ref_ratio, m_probtype, m_cur_time, m_dt, m_nstep, m_mu);"
    edit(
        "incflo.cpp",
        [
            ("#include <incflo.H>", "#include <incflo.H>\n#include <ns_case.H>"),
            (
                "    ReadParameters();",
                (
                    "    ReadParameters();\n"
                    "    ns_case::validate(m_fixed_dt, m_regrid_int, m_constant_density, m_ro_0, m_mu);"
                ),
            ),
            (
                "        if (m_check_int > 0) { WriteCheckPointFile(); }",
                "        "
                + call
                + "\n        if (m_check_int > 0) { WriteCheckPointFile(); }",
            ),
            (
                "        m_cur_time += m_dt;",
                "        m_cur_time += m_dt;\n        " + call,
            ),
            (
                "        if (writeNow())",
                "        if (ns_case::options().plot_times.empty() ? writeNow() : ns_case::plot_due(m_cur_time))",
            ),
            (
                "m_cur_time >= m_stop_time - (1.e-12 * m_dt)",
                "ns_case::reached_end(m_cur_time, m_stop_time, m_dt)",
                2,
            ),
        ],
    )
    # Preserve unchanged mtimes so reconfiguration does not rebuild all incflo.
    for relative, contents in generated.items():
        path = destination / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists() or path.read_bytes() != contents:
            path.write_bytes(contents)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    generate(args.source, args.destination)
