"""Build a Gazebo world containing the hand, with joint control enabled.

`gz sim file.sdf` loads a bare model, but Gazebo will not accept joint commands
unless the world contains a `JointController` system - the `/joint/<name>/cmd`
topics only exist once it does. So this wraps the converted hand model in a
world that adds:

  * physics with real gravity and a ground plane
  * a `JointController` (cmd_pose + PID) so each finger can be driven by angle
  * a camera and a light, so the GUI opens on something you can actually see

Optionally a forearm/wrist is prepended, so the hand is attached to an arm
rather than floating in space.
"""

from __future__ import annotations

from xml.etree import ElementTree as ET

from .robot_hand import FINGERS
from .urdf_export import JOINT_SUFFIX, build_urdf

# URDF -> SDF via gz is done by the launch script; this module works from the
# hand model XML directly so it can wrap it in a <world>.

WORLD_NAME = "robohand_world"
WRIST_JOINT_NAME = "wrist_pitch"


def _sdf_model_from_urdf(urdf_text: str, name: str) -> str:
    """Hand the URDF to gz's own converter, returning the <model> element."""
    import subprocess
    import tempfile
    import os

    with tempfile.NamedTemporaryFile("w", suffix=".urdf", delete=False) as fh:
        fh.write('<?xml version="1.0"?>\n' + urdf_text)
        path = fh.name
    try:
        out = subprocess.run(
            ["gz", "sdf", "-p", path], capture_output=True, text=True, timeout=120
        ).stdout
    finally:
        os.unlink(path)
    root = ET.fromstring(out)
    for model in root.iter("model"):
        model.set("name", name)
        return ET.tostring(model, encoding="unicode")
    raise RuntimeError("gz sdf -p produced no <model>; is gz on PATH?")


def _mount_hand(hand_xml: str, pose: str) -> str:
    """Set the hand model's root <pose>.

    Done through the XML tree, not a string replace: the converted model has
    several <pose> elements and only the first (the model's own) is the mount.
    """
    root = ET.fromstring(f"<root>{hand_xml}</root>")
    model = root.find("model")
    if model is None:
        return hand_xml
    p = model.find("pose")
    if p is None:
        p = ET.Element("pose")
        model.insert(0, p)
    p.text = pose
    return ET.tostring(model, encoding="unicode")


def _arm_links() -> tuple[str, str]:
    """Forearm + wrist links, and the joints that chain them onto the hand.

    These are injected *into the hand's own model* on purpose. An SDF joint can
    only ever join two frames inside the same model, so a separate forearm model
    cannot be attached with a joint - and nesting the hand inside an arm model
    would rename the joint-controller topics. Keeping one model means the
    15 finger joints keep clean `/model/robohand_Right/joint/...` topics and the
    forearm simply hangs off the bottom.

    Chain:  forearm_link --(wrist_pitch, revolute)--> wrist_link
                                     --(fixed)--> base_link (the hand)
    """
    links = """
      <link name="forearm_link">
        <pose>0 -0.245 0 0 0 0</pose>
        <inertial>
          <pose>0 0 0 0 0 0</pose>
          <mass>1.6</mass>
          <inertia>
            <ixx>0.006889</ixx><ixy>0</ixy><ixz>0</ixz>
            <iyy>0.006889</iyy><iyz>0</iyz><izz>0.000871</izz>
          </inertia>
        </inertial>
        <collision name="forearm_collision">
          <geometry><cylinder><radius>0.033</radius><length>0.22</length></cylinder></geometry>
        </collision>
        <visual name="forearm_visual">
          <geometry><cylinder><radius>0.033</radius><length>0.22</length></cylinder></geometry>
          <material>
            <ambient>0.30 0.33 0.38 1</ambient>
            <diffuse>0.38 0.42 0.48 1</diffuse>
            <specular>0.28 0.30 0.34 1</specular>
          </material>
        </visual>
      </link>
      <link name="wrist_link">
        <!-- Pose must be identity: placement of a jointed link comes from its
             joint's <pose>, which is relative to the parent. Giving it an
             absolute pose here too puts the link in two places at once. -->
        <pose>0 0 0 0 0 0</pose>
        <inertial>
          <mass>0.22</mass>
          <inertia>
            <ixx>0.000116</ixx><ixy>0</ixy><ixz>0</ixz>
            <iyy>0.000116</iyy><iyz>0</iyz><izz>0.000116</izz>
          </inertia>
        </inertial>
        <collision name="wrist_collision">
          <geometry><sphere><radius>0.028</radius></sphere></geometry>
        </collision>
        <visual name="wrist_visual">
          <geometry><sphere><radius>0.028</radius></sphere></geometry>
          <material>
            <ambient>0.42 0.45 0.50 1</ambient>
            <diffuse>0.52 0.56 0.62 1</diffuse>
          </material>
        </visual>
      </link>
"""
    joints = """
      <joint name="wrist_pitch" type="revolute">
        <parent>forearm_link</parent>
        <child>wrist_link</child>
        <pose>0 -0.128 0 0 0 0</pose>
        <axis>
          <xyz>1 0 0</xyz>
          <!-- limit and dynamics belong INSIDE <axis> per the SDF spec; as
               direct children of <joint> sdformat drops them with a warning. -->
          <limit>
            <lower>-1.15</lower><upper>1.15</upper>
            <effort>90</effort><velocity>2.5</velocity>
          </limit>
          <dynamics><damping>0.4</damping><friction>0.2</friction></dynamics>
        </axis>
      </joint>
      <joint name="wrist_to_hand" type="fixed">
        <parent>wrist_link</parent>
        <child>base_link</child>
        <pose>0 0 0 0 0 0</pose>
      </joint>
"""
    return links, joints


def _inject_arm(hand_xml: str) -> str:
    """Add the forearm links and their joints into the hand's model."""
    root = ET.fromstring(f"<root>{hand_xml}</root>")
    model = root.find("model")
    if model is None:
        return hand_xml
    links_xml, joints_xml = _arm_links()
    frag = ET.fromstring(f"<root>{links_xml}{joints_xml}</root>")
    for child in list(frag):
        model.append(child)
    return ET.tostring(model, encoding="unicode")


def _inject_controllers(hand_xml: str, controllers: list[str]) -> str:
    """Nest the JointController plugins inside the hand's <model> element."""
    root = ET.fromstring(f"<root>{hand_xml}</root>")
    model = root.find("model")
    if model is None:
        return hand_xml
    frag = ET.fromstring(f"<root>{''.join(controllers)}</root>")
    for child in list(frag):
        model.append(child)
    return ET.tostring(model, encoding="unicode")


def build_world(
    hand: str = "Right",
    model_name: str = "robohand_Right",
    with_arm: bool = True,
    world_name: str = WORLD_NAME,
) -> str:
    """Return a complete .sdf world with the hand (and optional forearm)."""
    hand_xml = _sdf_model_from_urdf(build_urdf(hand), model_name)
    if with_arm:
        hand_xml = _inject_arm(hand_xml)

    # NOTE 1: an SDF <plugin>'s `name` attribute must be the *registered class*
    # name - gz-sim's SystemLoader looks the plugin up by that name, so a unique
    # per-joint id like "joint_controller_index_mcp" fails with "library does
    # not contain requested plugin". Repeat the class name and let <joint_name>
    # distinguish the instances.
    # JointPositionController builds its command topic as
    #   /model/<model>/joint/<joint_name>/<joint_index>/cmd_pos
    # <joint_index> is NOT a free id: it indexes the joints whose name matches
    # <joint_name>, and each of ours has exactly one, so it must stay 0. The
    # joint *name* is what keeps the topics apart.
    controllers = []
    joint_index: dict[str, int] = {}

    def _controller(joint_name: str, gains: dict[str, str]) -> str:
        joint_index[joint_name] = 0
        pid = "\n".join(f"      <{k}>{v}</{k}>" for k, v in gains.items())
        return f"""      <plugin filename="gz-sim-joint-position-controller-system"
             name="gz::sim::systems::JointPositionController">
      <joint_name>{joint_name}</joint_name>
      <joint_index>{joint_index[joint_name]}</joint_index>
{pid}
    </plugin>"""

    for finger in ("thumb", "index", "middle", "ring", "pinky"):
        for suffix in JOINT_SUFFIX:
            controllers.append(_controller(
                f"{finger}_{suffix}",
                {"p_gain": 16, "i_gain": 0.3, "d_gain": 0.4,
                 "i_max": 4.0, "i_min": -4.0,
                 "cmd_max": 14.0, "cmd_min": -14.0},
            ))

    if with_arm:
        controllers.append(_controller(
            WRIST_JOINT_NAME,
            {"p_gain": 240, "i_gain": 30, "d_gain": 30,
             "i_max": 300, "i_min": -300,
             "cmd_max": 260, "cmd_min": -260},
        ))

    ground = """
    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry><plane><normal>0 0 1</normal><size>100 100</size></plane></geometry>
        </collision>
        <visual name="visual">
          <geometry><plane><normal>0 0 1</normal><size>100 100</size></plane></geometry>
          <material>
            <ambient>0.16 0.17 0.19 1</ambient>
            <diffuse>0.20 0.21 0.24 1</diffuse>
          </material>
        </visual>
      </link>
    </model>
    <light name="sun" type="directional">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 2 0 0 0</pose>
      <diffuse>0.9 0.9 0.9 1</diffuse>
      <specular>0.3 0.3 0.3 1</specular>
      <direction>-0.5 0.4 -0.8</direction>
    </light>
    <light name="fill" type="directional">
      <pose>0 0 2 0 0 0</pose>
      <diffuse>0.35 0.38 0.45 1</diffuse>
      <direction>0.6 -0.3 -0.7</direction>
    </light>
    <gui>
      <camera name="user_camera">
        <pose>-0.30 -0.20 0.20 0 0.30 0.55</pose>
        <view_controller>orbit</view_controller>
        <projection>
          <fov>1.05</fov><near>0.02</near><far>60</far>
        </projection>
      </camera>
    </gui>
"""

    # The controllers belong to the hand model, not to the world.
    hand_xml = _inject_controllers(hand_xml, controllers)

    return f"""<?xml version="1.0"?>
<sdf version="1.9">
  <world name="{world_name}">
    <physics name="1ms" type="dart">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
    </physics>
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
    <plugin filename="gz-sim-contact-system" name="gz::sim::systems::Contact"/>
{ground}
{hand_xml}
  </world>
</sdf>
"""


def joint_limits(hand: str = "Right") -> dict[str, tuple[float, float]]:
    """(lower, upper) per joint, parsed from the rig itself."""
    from .urdf_export import joint_table

    return {name: (lo, hi) for name, lo, hi, _ in joint_table()}


def command_topics(
    model: str = "robohand_Right",
    hand: str = "Right",
    with_arm: bool = True,
) -> dict[str, str]:
    """joint name -> gz-transport command topic, matching build_world().

    JointPositionController namespaces each command as
    ``/model/<model>/joint/<joint_name>/0/cmd_pos``. The trailing index is the
    controller's ``<joint_index>``, which indexes same-named joints; ours have
    exactly one each, so it is always 0.
    """
    names = [f"{f}_{s}" for f in FINGERS for s in JOINT_SUFFIX]
    if with_arm:
        names.append(WRIST_JOINT_NAME)
    return {n: f"/model/{model}/joint/{n}/0/cmd_pos" for n in names}


def cmd_topic(model: str, joint: str) -> str:
    return command_topics(model)[joint]


def world_file_name(hand: str = "Right", with_arm: bool = True) -> str:
    return f"robohand_{hand.lower()}{'_arm' if with_arm else ''}_world.sdf"
