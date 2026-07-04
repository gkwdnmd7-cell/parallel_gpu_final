#!/usr/bin/env python3
"""Convert JacobiSetComputation *_jacobi.txt to legacy VTK polydata (.vtk)."""

import sys

try:
    import vtk
except ImportError:
    print("缺少 vtk 模块。WSL 安装: sudo apt install python3-vtk9")
    print("或: pip install vtk")
    sys.exit(1)


def convert(infile: str) -> str:
    outfile = infile[:-4] + ".vtk" if infile.endswith(".txt") else infile + ".vtk"

    with open(infile, "r", encoding="utf-8") as f:
        f.readline()  # JacobiSet
        ne = int(f.readline().strip())

        points = vtk.vtkPoints()
        lines = vtk.vtkCellArray()

        for _ in range(ne):
            t = f.readline().split()
            # v1 idx, x,y,z, v2 idx, x,y,z
            points.InsertNextPoint(float(t[1]), float(t[2]), float(t[3]))
            points.InsertNextPoint(float(t[5]), float(t[6]), float(t[7]))

            pl = vtk.vtkPolyLine()
            pl.GetPointIds().SetNumberOfIds(2)
            pl.GetPointIds().SetId(0, points.GetNumberOfPoints() - 2)
            pl.GetPointIds().SetId(1, points.GetNumberOfPoints() - 1)
            lines.InsertNextCell(pl)

    poly = vtk.vtkPolyData()
    poly.SetPoints(points)
    poly.SetLines(lines)

    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(outfile)
    writer.SetInputData(poly)
    writer.SetFileTypeToBinary()
    writer.Write()
    return outfile


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <mesh_jacobi.txt>")
        sys.exit(1)
    out = convert(sys.argv[1])
    print(f"Wrote {out}")
