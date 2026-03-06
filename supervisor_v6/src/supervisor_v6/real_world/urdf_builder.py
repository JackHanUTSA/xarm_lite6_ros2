from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

from .world_reconstruction import WorldReconstructionEngine
from .command_motion_dataset import CommandMotionDataset


class URDFBuilder:
    """Generates a URDF from reconstructed meshes + observed joint limits."""

    def __init__(
        self,
        reconstruction: WorldReconstructionEngine,
        dataset: CommandMotionDataset,
        vlm_kinematics: dict,
        output_dir: str = "./robot_assets/",
        robot_name: str = "reconstructed_arm",
    ):
        self.recon = reconstruction
        self.dataset = dataset
        self.vlm_kin = vlm_kinematics
        self.out = Path(output_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.name = robot_name

    def build(self) -> str:
        joint_stats = self.dataset.joint_limit_stats()
        dof = len(joint_stats["min"])

        root = ET.Element("robot", name=self.name)
        ET.SubElement(root, "link", name="world")

        prev_link = "world"
        for i in range(dof):
            link_name = f"link_{i}"
            joint_name = f"joint_{i}"

            ET.SubElement(root, "link", name=link_name)
            joint_el = ET.SubElement(root, "joint", name=joint_name, type="revolute")
            ET.SubElement(joint_el, "parent", link=prev_link)
            ET.SubElement(joint_el, "child", link=link_name)
            ET.SubElement(joint_el, "origin", xyz="0 0 0.1", rpy="0 0 0")
            ET.SubElement(joint_el, "axis", xyz="0 0 1")

            lo = float(joint_stats["min"][i])
            hi = float(joint_stats["max"][i])
            margin = abs(hi - lo) * 0.05
            ET.SubElement(
                joint_el,
                "limit",
                lower=f"{lo - margin:.4f}",
                upper=f"{hi + margin:.4f}",
                effort="87",
                velocity="2.175",
            )
            prev_link = link_name

        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ")
        urdf_path = self.out / f"{self.name}.urdf"
        tree.write(str(urdf_path), xml_declaration=True, encoding="unicode")

        meta = {
            "robot_name": self.name,
            "dof": dof,
            "joint_limits": joint_stats,
            "vlm_kinematics": self.vlm_kin,
            "urdf_path": str(urdf_path),
        }
        urdf_path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return str(urdf_path)
