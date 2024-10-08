from typing import Dict, Optional, List, Tuple
import gymnasium as gym
from gymnasium import spaces
from collections import defaultdict
from enum import IntEnum
import random
import numpy as np
import math

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

from homegrid.base import (
  MiniGridEnv, Grid,
  Storage, Inanimate, Baby, Pickable
)
from homegrid.layout import ThreeRoom, CANS, TRASH, room2name


class HomeGridBase(MiniGridEnv):

  class Actions(IntEnum):
    left = 0
    right = 1
    up = 2
    down = 3
    # item actions
    pickup = 4
    drop = 5
    # storage actions
    get = 6
    pedal = 7
    grasp = 8
    lift = 9

  ac2dir = {
    Actions.right: 0,
    Actions.down: 1,
    Actions.left: 2,
    Actions.up: 3,
  }

  def __init__(self,
        layout=ThreeRoom,
        num_trashcans=2,
        num_trashobjs=2,
        view_size=3,
        max_steps=29,
        p_teleport=0.05,
        max_objects=4,
        p_unsafe=0.0,
        fixed_state=None,
        n_babies=1,
        ):
    self.layout = layout()
    self.textures = self.layout.textures
    super().__init__(
      width=self.layout.width,
      height=self.layout.height,
      render_mode="rgb_array",
      agent_view_size=view_size,
      max_steps=max_steps,
    )
    self.actions = HomeGridBase.Actions

    # ORIGINAL
    # Full use of homegrid actions
    #self.action_space = spaces.Discrete(len(self.actions))

    # EDIT - Agent only moves up down left right
    self.action_space = spaces.Discrete(4)


    self.num_trashcans = num_trashcans
    self.num_trashobjs = num_trashobjs
    self.p_teleport = p_teleport
    self.max_objects = max_objects
    self.p_unsafe = p_unsafe
    self.fixed_state = fixed_state
    self.n_babies=n_babies

  @property
  def step_cnt(self):
      return self.step_count

  # def init_from_state(self, state):
  #   """Initialize the env from a symbolic state."""
  #   self._create_layout(self.width, self.height)
  #   # place agent
  #   self.agent_pos = state["agent"]["pos"]
  #   self.agent_dir = state["agent"]["dir"]
  #   self.objs = []
  #   # place objects with appropriate state
  #   for ob in state["objects"]:
  #     if ob["pos"] == (-1, -1):
  #       print(f"Skipping carried object {ob['name']}")
  #       continue
  #     if ob["type"] == "Storage":
  #       pfx = ob["name"].replace(" ", "_")
  #       obj = Storage(
  #         name=ob["name"],
  #         textures={
  #           "open": self.textures[f"{pfx}_open"],
  #           "closed": self.textures[f"{pfx}_closed"]},
  #         state=ob["state"],
  #         action=ob["action"],
  #         contains=ob["contains"],
  #       )
  #     elif ob["type"] == "Pickable":
  #       obj = Pickable(
  #         name=ob["name"],
  #         texture=self.textures[ob["name"]],
  #         invisible=ob["invisible"],
  #       )
  #     else:
  #       raise NotImplementedError("Obj type {ob['type']}")
  #     self.objs.append(obj)
  #     self.place_obj(obj, top=ob["pos"],
  #                    size=(1,1), max_tries=1)

  def _add_cans_to_house(self):
    cans = random.sample(CANS, self.num_trashcans)
    #poss = random.sample(self.layout.valid_poss["can"], self.num_trashcans)
    can_objs = []
    for i, can in enumerate(cans):
      obj = Storage(can, {
        "open": self.textures[f"{can}_open"],
        "closed": self.textures[f"{can}_closed"]},
        # Make one of the cans irreversibly broken
        reset_broken_after=200 if i == 0 else 5
      )
      can_objs.append(obj)
      
      #pos = self.place_obj(obj, top=poss[i], size=(1,1), max_tries=5)
    pos = self.place_obj(can_objs, self.layout.valid_poss["can"], max_tries=20)
    self.objs = self.objs + can_objs
  

  def place_at_mid_location(self, obj, A, B, valid_positions, max_tries=math.inf, not_allowed=[]):

    A, B = np.array(A), np.array(B)
    val = np.array(valid_positions)

    assert A.shape == B.shape == (2,)

    # Pick a ramdom point along A to B vector
    random_value = np.random.uniform()
    random_point = A + random_value*(B - A)

    # Sort valid locations by distance from point
    sq_dist = np.sum((val - random_point)**2, axis=1)
    sort_idxs = sq_dist.argsort()

    # Try the positions in order of proximity to chosen point
    for num_tries, pos_idx in enumerate(sort_idxs):
      
      if num_tries > max_tries:
        raise RecursionError("rejection sampling failed in place_obj")
      
      pos = tuple(valid_positions[pos_idx])
                
      # SPECIFIC EDITS FOR LLM-ALIGNED-RL
      if pos in not_allowed:
        continue

      # Don't place the object on top of another object
      if self.grid.get(*pos) is not None and not self.grid.get(*pos).can_overlap():
        continue

      # Don't place the object where the agent is
      if np.array_equal(pos, self.agent_pos):
        continue
      
      break

    self.grid.set(pos[0], pos[1], obj)

    if obj is not None:
        obj.init_pos = pos
        obj.cur_pos = pos

    return pos


  def buffer(self, not_allowed, size=1):
    # Expand around 'not allowed' entries to create a buffer of non-allowed squares
    full_list = []
    for i, j in not_allowed:
      full_list += [(i + del_i, j + del_j) for del_i in range(-size, size+1) for del_j in range(-size, size+1)]
    not_allowed = full_list
    return not_allowed


  def _add_cat_to_house(self, not_allowed=[]):
    obj = Baby()
    #poss = random.sample(self.layout.valid_poss["agent_start"], 1)
    # pos = self.place_obj(obj, self.layout.valid_poss["agent_start"], max_tries=20, not_allowed=not_allowed)
    
    pos = self.place_at_mid_location(
      obj,
      self.agent_pos,
      self.fruit_location,
      self.layout.valid_poss["agent_start"],
      not_allowed=not_allowed
    )
    self.cat_location = pos
    self.objs.append(obj)


  def _add_fruit_to_house(self, not_allowed=[]):
    #poss = random.sample(self.layout.valid_poss["obj"], 1)
    obj = Pickable("fruit", self.textures["fruit"])
    pos = self.place_obj(obj, self.layout.valid_poss["obj"], max_tries=20, not_allowed=not_allowed)
    self.fruit_location = tuple(pos[0])
    self.objs.append(obj)

  # def _add_objs_to_house(self):
  #   trash_objs = random.sample(TRASH, self.num_trashobjs)
  #   poss = random.sample(self.layout.valid_poss["obj"], self.num_trashobjs)
  #   trashobj_objs = []
  #   for i, trash in enumerate(trash_objs):
  #     obj = Pickable(trash, self.textures[trash])
  #     pos = self.place_obj(obj, top=poss[i], size=(1,1), max_tries=5)
  #     trashobj_objs.append(obj)
  #     self.objs.append(obj)

  def _gen_grid(self, width, height):
    if self.fixed_state:
      print("Initializing from fixed state")
      self.init_from_state(self.fixed_state)
      return
    regenerate = True
    while regenerate:
      self._create_layout(width, height)
      regenerate = False

      self.objs = []
      self.goal = {"obj": None, "can": None}
      # Place objects
      #self._add_cans_to_house()
      self._add_fruit_to_house()
      
    # Place agent
    agent_not_allowed = self.buffer([self.fruit_location], size=3)
    self.agent_pos = self.place_agent(self.layout.valid_poss["agent_start"], max_tries = 20, not_allowed=agent_not_allowed)
    cat_not_allowed = self.buffer([self.agent_pos, self.fruit_location], size=1)
    self._add_cat_to_house(not_allowed=cat_not_allowed)


  def _create_layout(self, width, height):
    # Create grid with surrounding walls
    self.grid = Grid(width, height)
    self.layout.populate(self.grid)
    self.room_to_cells = self.layout.room_to_cells
    self.cell_to_room = self.layout.cell_to_room

  def _maybe_teleport(self):
    if np.random.random() > self.p_teleport:
      return False
    objs = [o for o in self.objs if isinstance(o, Pickable) and o.cur_pos[0] !=
        -1 and not o.invisible]
    if len(objs) == 0:
      print(self.objs)
      print([o.cur_pos for o in self.objs])
      print(self.all_events)
      return False
    obj = random.choice(objs)
    # Choose a random new location with no object to place this
    poss = random.choice([
      pos for pos in self.layout.valid_poss["obj"] \
      if pos != obj.cur_pos and self.grid.get(*pos) is None \
      and pos != self.agent_pos])
    self.grid.set(*obj.cur_pos, None)
    obj.cur_pos = poss
    self.grid.set(*poss, obj)
    return obj

  def _maybe_spawn(self):
    new_objs = [t for t in TRASH if t not in \
        [o.name for o in self.objs]]
    if np.random.rand() < 0.1 * len(new_objs):
      trash = random.choice(new_objs)
      obj = Pickable(trash, self.textures[trash], invisible=True)
      poss = random.choice([
        pos for pos in self.layout.valid_poss["obj"] \
        if pos != obj.cur_pos and self.grid.get(*pos) is None \
        and pos != self.agent_pos])
      self.place_obj(obj, top=poss, size=(1,1), max_tries=5)
      self.objs.append(obj)
      return obj
    return None

  def _maybe_unsafe(self):
    if len(self.unsafe_poss) == 0 and np.random.rand() < self.p_unsafe:
      can = random.choice([o for o in self.objs if isinstance(o, Storage)])
      self.unsafe_poss = set()
      self.unsafe_name = can.name
      for x in [can.cur_pos[0] - 1, can.cur_pos[0], can.cur_pos[0] + 1]:
        for y in [can.cur_pos[1] - 1, can.cur_pos[1], can.cur_pos[1] + 1]:
          self.unsafe_poss.add((x,y))
      self.unsafe_end = self.step_count + random.randint(1, 10)
      return can
    return None

  def reset(self, *, seed=None, options=None):
    self.prev_action = "Reset"
    obs, info = super().reset(seed=seed, options=options)
    # All events in the episode so far
    self.all_events = []
    self.step_count = 0
    self.unsafe_poss = {}
    self.unsafe_end = -1
    self.unsafe_name = None
    info = {
        "symbolic_state": self.get_full_symbolic_state(),
        "events": []
    }
    self.cat_squashed = False

    self.reduced_grids = []
    self.add_reduced_grid()

    return obs, info
  

  def add_reduced_grid(self):
    reduced_grid = self.binary_grid[:, 1:-1, :-1].copy()
    self.reduced_grids.append(reduced_grid)


  def step(self, action):
    self.step_count += 1

    reward = 0
    terminated = False
    truncated = False
    success = None
    events = []

    # Step all the object states
    for obj in self.objs:
      obj.tick()

    # Get the position in front of the agent
    fwd_pos = self.front_pos

    # Get the contents of the cell in front of the agent
    fwd_cell = self.grid.get(*fwd_pos)
    fwd_floor = self.grid.get_floor(*fwd_pos)

    if action == self.actions.left or \
        action == self.actions.right or \
        action == self.actions.up or \
        action == self.actions.down:
      self.agent_dir = HomeGridBase.ac2dir[action]
     # Get the position in front of the agent after turning
      fwd_pos = self.front_pos

      # Get the contents of the cell in front of the agent
      fwd_cell = self.grid.get(*fwd_pos)
      fwd_floor = self.grid.get_floor(*fwd_pos)

      if (fwd_cell is None or fwd_cell.agent_can_overlap()) and \
          (fwd_floor is None or fwd_floor.agent_can_overlap()):
        if isinstance(fwd_cell, Baby):
          fwd_cell.squash()
          self.cat_squashed = True
        self.agent_pos = tuple(fwd_pos)

    # EDIT - OTHER ACTIONS DISABLED FOR FIRST DRAFT

    # # Pick up an object
    # if action == self.actions.pickup:
    #   if isinstance(fwd_cell, Pickable) and fwd_cell.can_pickup():
    #     if self.carrying is None:
    #       self.carrying = fwd_cell
    #       self.carrying.cur_pos = (-1, -1)
    #       self.grid.set(fwd_pos[0], fwd_pos[1], None)

    # # Drop an object
    # elif action == self.actions.drop:
    #   if not fwd_cell and self.carrying:
    #     self.grid.set(fwd_pos[0], fwd_pos[1], self.carrying)
    #     self.carrying.cur_pos = tuple(fwd_pos)
    #     self.carrying = None
    #   elif isinstance(fwd_cell, Storage) and self.carrying:
    #     succeeded = fwd_cell.interact(self.actions(action).name,
    #                                   obj=self.carrying)
    #     if succeeded:
    #       self.carrying = None

    # elif action == self.actions.get:
    #   if not self.carrying and isinstance(fwd_cell, Storage):
    #     obj = fwd_cell.interact(self.actions(action).name)
    #     if obj:
    #       self.carrying = obj
    #       self.carrying.cur_pos = (-1, -1)

    # elif action == self.actions.pedal or action == self.actions.grasp or \
    #     action == self.actions.lift:
    #   if isinstance(fwd_cell, Storage):
    #     succeeded = fwd_cell.interact(self.actions(action).name)

    else:
      raise ValueError(f"Unknown action: {action}")

    if self.step_count >= self.max_steps:
      truncated = True
    if (terminated or truncated) and success is None:
      success = False

    if self.render_mode == "human":
      self.render_with_text()
    obs = self.gen_obs()

    # For rendering purposes
    self.prev_action = HomeGridBase.Actions(action).name
    if success:
      self.done_condition = "success"
    elif truncated:
      self.done_condition = "truncated"

    # Random events
    # if self.agent_pos in self.unsafe_poss:
    #   terminated = True
    #   reward = -1

    # if len(self.unsafe_poss) > 0 and self.step_count == self.unsafe_end:
    #   self.unsafe_poss = {}
    #   self.unsafe_name = None
    #   self.unsafe_end = -1
    #   events.append({
    #     "type": "termination",
    #     "description": f"i cleaned the spill",
    #     })
    # elif len(self.unsafe_poss) == 0:
    #   obj = self._maybe_unsafe()
    #   if obj:
    #     events.append({
    #       "type": "termination",
    #       "description": f"spill near the {self.unsafe_name}",
    #       })
    # else:
    #     events.append({
    #       "type": "termination",
    #       "description": f"spill near the {self.unsafe_name}",
    #       })
    # obj = self._maybe_teleport()
    # if obj:
    #   room_code = self.cell_to_room[obj.cur_pos]
    #   room_name = room2name[room_code]
    #   events.append({
    #     "type": "future", "obj": obj,
    #     "room": room_code,
    #     "description": f"i moved the {obj.name} to the {room_name}"
    #     })
    # obj = self._maybe_spawn()
    # if obj:
    #   room_code = self.cell_to_room[obj.cur_pos]
    #   room_name = room2name[room_code]
    #   events.append({
    #     "type": "future", "obj": obj,
    #     "description": f"there will be {obj.name} in the {room_name} later"
    #     })


    self.all_events.append(events)
    info = {
      "success": success,
      "action": action,
      "symbolic_state": self.get_full_symbolic_state(),
      "events": events,
      "all_events": self.all_events,
    }

    #self.save_environment_image(f'../../test_rgb_out/{self.step_count}.png')

    #obs = self.agent_view_binary_grid()
    
    obs = self.update_binary_grid([
        self.agent_pos,
        None,
        None,
        None
    ])
    obs = obs*255

    self.add_reduced_grid()

    terminated = self.agent_facing_fruit

    reward = self.face_fruit_reward(truncated)

    return obs, reward, terminated, truncated, info


  @property
  def agent_facing_fruit(self):
    return tuple(self.front_pos) == self.fruit_location

  def face_fruit_reward(self, truncated):
    if self.agent_facing_fruit:
      return self._reward()
    if truncated:
      return self._reward()/2
    return 0
  



  def save_environment_image(self, savefile):
    img = self.get_frame(highlight=False)
    img = Image.fromarray(img)
    img.save(savefile)
    print(f"\n\n\nIMAGE SAVED TO {savefile}\n\n\n")
    


  def get_full_symbolic_state(self) -> Dict:
    fwd_pos = self.front_pos
    fwd_cell = self.grid.get(*fwd_pos)
    if isinstance(fwd_cell, Pickable) or isinstance(fwd_cell, Storage):
      front_obj = fwd_cell.name
    else:
      front_obj = None

    state = {
      "step": self.step_count,
      "agent": {
        "pos": self.agent_pos,
        "room": self.cell_to_room[self.agent_pos] if self.agent_pos in self.cell_to_room else None,
        "dir": self.agent_dir,
        "carrying": self.carrying.name if self.carrying else None
      },
      "objects": [
        {
          "name": obj.name,
          "type": obj.__class__.__name__,
          "pos": obj.cur_pos,
          "room": self.cell_to_room[obj.cur_pos] if (obj.cur_pos[0] != -1 and obj.cur_pos in self.cell_to_room) else None,
          "state": getattr(obj, "state", None),
          "action": getattr(obj, "action", None),
          "invisible": getattr(obj, "invisible", None),
          "contains": [contained_obj.name for contained_obj in obj.contains] if isinstance(obj, Storage) else None,
        } for obj in self.objs
      ],
      "front_obj": front_obj,
      "unsafe": {
        "name": self.unsafe_name,
        "poss": self.unsafe_poss,
        "end": self.unsafe_end,
        }
    }

    return state
  

  def render_with_text(self, text):
    img = self._env.render(mode="rgb_array")
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), text, (0, 0, 0))
    draw.text((0, 45), "Action: {}".format(self._env.prev_action), (0, 0, 0))
    img = np.asarray(img)
    return img
