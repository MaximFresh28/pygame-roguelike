#!/usr/bin/env python3
"""
Пошаговый roguelike на Pygame.

Каждый уровень подземелья генерируется заново: комнаты, коридоры, монстры, предметы.
Видимость считается лучами (туман войны), а монстры преследуют игрока
по карте расстояний, построенной обходом в ширину (BFS).

Запуск:  python roguelike.py            случайное подземелье
         python roguelike.py --seed 42  одно и то же подземелье при каждом запуске
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
from typing import Deque, Dict, Iterator, List, Optional, Set, Tuple

os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
import pygame  # noqa: E402

Point = Tuple[int, int]
RGB = Tuple[int, int, int]

# ═══════════════════════════════════════════════════════════════════════════
#  Настройки
# ═══════════════════════════════════════════════════════════════════════════

TILE = 18
MAP_W, MAP_H = 48, 30
PANEL_H = 150
SCREEN_W = MAP_W * TILE
SCREEN_H = MAP_H * TILE + PANEL_H
FPS = 60

FOV_RADIUS = 8
MAX_ROOMS = 14
ROOM_MIN, ROOM_MAX = 5, 11
MONSTER_MEMORY = 6          # сколько ходов монстр помнит игрока, потеряв его из виду
CHASE_RANGE = 30            # дальше этого карта расстояний не строится
REGEN_EVERY = 10            # +1 здоровья каждые N ходов
LOG_SIZE = 6

DIRS4 = ((0, -1), (0, 1), (-1, 0), (1, 0))


class Color:
    BG = (12, 11, 18)
    WALL_LIT = (92, 82, 70)
    WALL_LIT_EDGE = (70, 62, 53)
    WALL_DARK = (36, 33, 31)
    FLOOR_LIT = (44, 42, 58)
    FLOOR_DARK = (21, 20, 29)
    FLOOR_DOT = (66, 63, 84)
    STAIRS = (240, 200, 80)
    STAIRS_DARK = (110, 92, 45)
    PLAYER = (255, 240, 170)
    PLAYER_GLOW = (78, 70, 48)
    PANEL = (20, 19, 30)
    PANEL_LINE = (52, 50, 76)
    TEXT = (225, 222, 240)
    MUTED = (130, 126, 160)
    HP = (196, 52, 62)
    HP_BG = (58, 24, 30)
    XP = (112, 92, 222)
    XP_BG = (34, 30, 60)
    GOOD = (120, 220, 130)
    BAD = (240, 100, 100)
    INFO = (150, 180, 255)
    LOOT = (240, 200, 80)


# ═══════════════════════════════════════════════════════════════════════════
#  Карта
# ═══════════════════════════════════════════════════════════════════════════

class Tile(Enum):
    WALL = auto()
    FLOOR = auto()
    STAIRS = auto()


@dataclass(frozen=True)
class Rect:
    x: int
    y: int
    w: int
    h: int

    @property
    def x2(self) -> int:
        return self.x + self.w - 1

    @property
    def y2(self) -> int:
        return self.y + self.h - 1

    @property
    def center(self) -> Point:
        return (self.x + self.x2) // 2, (self.y + self.y2) // 2

    def intersects(self, other: 'Rect', margin: int = 1) -> bool:
        """margin=1 оставляет между комнатами хотя бы одну клетку стены."""
        return (self.x - margin <= other.x2 and self.x2 + margin >= other.x
                and self.y - margin <= other.y2 and self.y2 + margin >= other.y)

    def random_point(self, rng: random.Random) -> Point:
        return rng.randint(self.x, self.x2), rng.randint(self.y, self.y2)


def bresenham(x0: int, y0: int, x1: int, y1: int) -> Iterator[Point]:
    """Клетки на отрезке между двумя точками — основа для лучей видимости."""
    dx, dy = abs(x1 - x0), -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        yield x0, y0
        if x0 == x1 and y0 == y1:
            return
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


class GameMap:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.tiles: List[List[Tile]] = [[Tile.WALL] * width for _ in range(height)]
        self.visible: Set[Point] = set()
        self.explored: Set[Point] = set()

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def is_walkable(self, x: int, y: int) -> bool:
        return self.in_bounds(x, y) and self.tiles[y][x] is not Tile.WALL

    # ── Генерация ────────────────────────────────────────────────────────

    def carve_room(self, room: Rect) -> None:
        for y in range(room.y, room.y2 + 1):
            for x in range(room.x, room.x2 + 1):
                self.tiles[y][x] = Tile.FLOOR

    def connect(self, a: Point, b: Point, rng: random.Random) -> None:
        """Г-образный коридор; случайный выбор изгиба делает карты разнообразнее."""
        (x1, y1), (x2, y2) = a, b
        if rng.random() < 0.5:
            self._carve_h(x1, x2, y1)
            self._carve_v(y1, y2, x2)
        else:
            self._carve_v(y1, y2, x1)
            self._carve_h(x1, x2, y2)

    def _carve_h(self, x1: int, x2: int, y: int) -> None:
        for x in range(min(x1, x2), max(x1, x2) + 1):
            self.tiles[y][x] = Tile.FLOOR

    def _carve_v(self, y1: int, y2: int, x: int) -> None:
        for y in range(min(y1, y2), max(y1, y2) + 1):
            self.tiles[y][x] = Tile.FLOOR

    # ── Видимость ────────────────────────────────────────────────────────

    def compute_fov(self, ox: int, oy: int, radius: int) -> None:
        """
        Лучи из позиции игрока к каждой клетке на границе квадрата радиуса r.
        Луч останавливается на первой стене. Затем подсвечиваются стены,
        примыкающие к видимому полу, — это закрывает «дыры» лучевого метода.
        """
        limit = radius * radius + radius           # чуть больше r² — круг выглядит ровнее
        visible: Set[Point] = {(ox, oy)}

        edge: List[Point] = []
        for t in range(-radius, radius + 1):
            edge += [(ox + t, oy - radius), (ox + t, oy + radius),
                     (ox - radius, oy + t), (ox + radius, oy + t)]

        for tx, ty in edge:
            for x, y in bresenham(ox, oy, tx, ty):
                if not self.in_bounds(x, y) or (x - ox) ** 2 + (y - oy) ** 2 > limit:
                    break
                visible.add((x, y))
                if self.tiles[y][x] is Tile.WALL:
                    break

        for x, y in list(visible):
            if self.tiles[y][x] is Tile.WALL:
                continue
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    nx, ny = x + dx, y + dy
                    if (self.in_bounds(nx, ny) and self.tiles[ny][nx] is Tile.WALL
                            and (nx - ox) ** 2 + (ny - oy) ** 2 <= limit):
                        visible.add((nx, ny))

        self.visible = visible
        self.explored |= visible

    # ── Поиск пути ───────────────────────────────────────────────────────

    def distance_map(self, start: Point, max_dist: int) -> Dict[Point, int]:
        """
        Расстояние от start до каждой достижимой клетки (BFS).
        Строится один раз за ход: каждому монстру достаточно шагнуть
        в соседнюю клетку с меньшим значением, чтобы идти к игроку
        кратчайшим путём. Это дешевле, чем отдельный поиск пути на каждого.
        """
        dist = {start: 0}
        queue: Deque[Point] = deque([start])
        while queue:
            x, y = queue.popleft()
            d = dist[(x, y)]
            if d >= max_dist:
                continue
            for dx, dy in DIRS4:
                nxt = (x + dx, y + dy)
                if nxt not in dist and self.is_walkable(*nxt):
                    dist[nxt] = d + 1
                    queue.append(nxt)
        return dist


def generate_dungeon(rng: random.Random) -> Tuple[GameMap, List[Rect]]:
    """
    Случайные непересекающиеся комнаты; каждая новая соединяется коридором
    с предыдущей, поэтому всё подземелье гарантированно связно.
    Игрок начинает в первой комнате, лестница — в центре последней.
    """
    while True:
        game_map = GameMap(MAP_W, MAP_H)
        rooms: List[Rect] = []
        for _ in range(MAX_ROOMS * 4):
            if len(rooms) >= MAX_ROOMS:
                break
            w = rng.randint(ROOM_MIN, ROOM_MAX)
            h = rng.randint(ROOM_MIN - 1, ROOM_MAX - 3)
            room = Rect(rng.randint(1, MAP_W - w - 1), rng.randint(1, MAP_H - h - 1), w, h)
            if any(room.intersects(other) for other in rooms):
                continue
            game_map.carve_room(room)
            if rooms:
                game_map.connect(rooms[-1].center, room.center, rng)
            rooms.append(room)
        if len(rooms) >= 3:
            break

    sx, sy = rooms[-1].center
    game_map.tiles[sy][sx] = Tile.STAIRS
    return game_map, rooms


# ═══════════════════════════════════════════════════════════════════════════
#  Существа и предметы
# ═══════════════════════════════════════════════════════════════════════════

class Entity:
    def __init__(self, x: int, y: int, name: str, glyph: str, color: RGB) -> None:
        self.x = x
        self.y = y
        self.name = name
        self.glyph = glyph
        self.color = color

    @property
    def pos(self) -> Point:
        return self.x, self.y


class Actor(Entity):
    def __init__(self, x: int, y: int, name: str, glyph: str, color: RGB,
                 max_hp: int, attack: int, defense: int) -> None:
        super().__init__(x, y, name, glyph, color)
        self.max_hp = max_hp
        self.hp = max_hp
        self.attack = attack
        self.defense = defense

    @property
    def alive(self) -> bool:
        return self.hp > 0

    def take_damage(self, amount: int) -> None:
        self.hp = max(0, self.hp - amount)

    def heal(self, amount: int) -> int:
        before = self.hp
        self.hp = min(self.max_hp, self.hp + amount)
        return self.hp - before


class Player(Actor):
    def __init__(self) -> None:
        super().__init__(0, 0, "Вы", "@", Color.PLAYER, max_hp=30, attack=5, defense=1)
        self.level = 1
        self.xp = 0
        self.potions = 1
        self.kills = 0

    @property
    def xp_to_next(self) -> int:
        return 25 * self.level

    def gain_xp(self, amount: int) -> List[str]:
        self.xp += amount
        messages = []
        while self.xp >= self.xp_to_next:
            self.xp -= self.xp_to_next
            self.level += 1
            self.max_hp += 6
            self.hp = self.max_hp
            self.attack += 1
            if self.level % 2 == 0:
                self.defense += 1
            messages.append(f"Новый уровень: {self.level}! Здоровье восстановлено.")
        return messages


@dataclass(frozen=True)
class MonsterKind:
    name: str
    glyph: str
    color: RGB
    hp: int
    attack: int
    defense: int
    xp: int
    min_depth: int
    weight: int


MONSTER_KINDS: Tuple[MonsterKind, ...] = (
    MonsterKind("Крыса", "r", (176, 144, 112), hp=4, attack=2, defense=0, xp=5, min_depth=1, weight=10),
    MonsterKind("Гоблин", "g", (112, 204, 92), hp=7, attack=3, defense=0, xp=10, min_depth=1, weight=8),
    MonsterKind("Орк", "o", (72, 164, 72), hp=12, attack=4, defense=1, xp=20, min_depth=2, weight=6),
    MonsterKind("Скелет", "s", (222, 222, 212), hp=10, attack=5, defense=2, xp=25, min_depth=3, weight=5),
    MonsterKind("Тролль", "T", (92, 172, 162), hp=22, attack=6, defense=2, xp=45, min_depth=4, weight=3),
    MonsterKind("Демон", "D", (232, 72, 62), hp=30, attack=8, defense=3, xp=80, min_depth=6, weight=2),
)


class Monster(Actor):
    def __init__(self, kind: MonsterKind, x: int, y: int, depth: int) -> None:
        bonus = max(0, depth - kind.min_depth)     # чем глубже, тем сильнее те же монстры
        super().__init__(x, y, kind.name, kind.glyph, kind.color,
                         max_hp=kind.hp + bonus * 2,
                         attack=kind.attack + bonus // 3,
                         defense=kind.defense + bonus // 4)
        self.xp_reward = kind.xp + bonus * 2
        self.alert = 0


@dataclass(frozen=True)
class ItemKind:
    key: str
    name: str
    glyph: str
    color: RGB
    weight: int


ITEM_KINDS: Tuple[ItemKind, ...] = (
    ItemKind("potion", "Зелье лечения", "!", (232, 82, 122), weight=60),
    ItemKind("sword", "Меч", "/", (182, 202, 232), weight=14),
    ItemKind("shield", "Щит", "]", (202, 172, 112), weight=14),
    ItemKind("amulet", "Амулет жизни", '"', (242, 122, 152), weight=12),
)


class Item(Entity):
    def __init__(self, kind: ItemKind, x: int, y: int) -> None:
        super().__init__(x, y, kind.name, kind.glyph, kind.color)
        self.kind = kind


# ═══════════════════════════════════════════════════════════════════════════
#  Игра — вся логика, без единого вызова pygame
# ═══════════════════════════════════════════════════════════════════════════

Action = Tuple  # ('move', dx, dy) | ('wait',) | ('drink',) | ('descend',) | ('restart',)


class Game:
    def __init__(self, seed: Optional[int] = None) -> None:
        self.rng = random.Random(seed)
        self.best_depth = 0
        self.new_game()

    def new_game(self) -> None:
        self.player = Player()
        self.depth = 0
        self.turn = 0
        self.dead = False
        self.killer: Optional[str] = None
        self.log: Deque[Tuple[str, RGB]] = deque(maxlen=LOG_SIZE)
        self.next_level()
        self.message("Вы спускаетесь в подземелье. Найдите лестницу «>».", Color.INFO)

    def message(self, text: str, color: RGB = Color.TEXT) -> None:
        self.log.append((text, color))

    # ── Уровни ───────────────────────────────────────────────────────────

    def next_level(self) -> None:
        self.depth += 1
        self.best_depth = max(self.best_depth, self.depth)
        self.map, rooms = generate_dungeon(self.rng)
        self.player.x, self.player.y = rooms[0].center
        self.monsters: List[Monster] = []
        self.items: List[Item] = []
        for room in rooms[1:]:                    # первая комната — безопасная
            self._populate(room)
        self._update_fov()

    def _populate(self, room: Rect) -> None:
        max_monsters = min(1 + self.depth // 2, 4)
        for _ in range(self.rng.randint(0, max_monsters)):
            spot = self._free_spot(room)
            if spot:
                self.monsters.append(Monster(self._pick_monster_kind(), *spot, self.depth))
        if self.rng.random() < 0.5:
            spot = self._free_spot(room)
            if spot:
                self.items.append(Item(self._pick_item_kind(), *spot))

    def _free_spot(self, room: Rect) -> Optional[Point]:
        for _ in range(10):
            x, y = room.random_point(self.rng)
            if self.map.tiles[y][x] is Tile.FLOOR and not self.actor_at(x, y) and not self.item_at(x, y):
                return x, y
        return None

    def _pick_monster_kind(self) -> MonsterKind:
        kinds = [k for k in MONSTER_KINDS if k.min_depth <= self.depth]
        return self.rng.choices(kinds, weights=[k.weight for k in kinds])[0]

    def _pick_item_kind(self) -> ItemKind:
        return self.rng.choices(ITEM_KINDS, weights=[k.weight for k in ITEM_KINDS])[0]

    # ── Запросы к миру ───────────────────────────────────────────────────

    def actor_at(self, x: int, y: int) -> Optional[Actor]:
        if self.player.pos == (x, y):
            return self.player
        for monster in self.monsters:
            if monster.pos == (x, y):
                return monster
        return None

    def item_at(self, x: int, y: int) -> Optional[Item]:
        for item in self.items:
            if item.pos == (x, y):
                return item
        return None

    @property
    def on_stairs(self) -> bool:
        return self.map.tiles[self.player.y][self.player.x] is Tile.STAIRS

    # ── Действия игрока ──────────────────────────────────────────────────

    def perform(self, action: Action) -> None:
        kind = action[0]
        if self.dead:
            if kind == 'restart':
                self.new_game()
            return

        if kind == 'move':
            spent = self._move_or_attack(action[1], action[2])
        elif kind == 'wait':
            spent = True
        elif kind == 'drink':
            spent = self._drink()
        elif kind == 'descend':
            self._descend()
            return
        else:
            return

        if spent:
            self._end_turn()

    def _move_or_attack(self, dx: int, dy: int) -> bool:
        p = self.player
        tx, ty = p.x + dx, p.y + dy
        target = self.actor_at(tx, ty)
        if isinstance(target, Monster):
            self._attack(p, target)
            return True
        if not self.map.is_walkable(tx, ty):
            return False                          # упёрлись в стену — ход не потрачен

        p.x, p.y = tx, ty
        item = self.item_at(tx, ty)
        if item:
            self._pick_up(item)
        if self.on_stairs:
            self.message("Здесь лестница вниз. Enter — спуститься.", Color.LOOT)
        return True

    def _attack(self, attacker: Actor, defender: Actor) -> None:
        damage = max(1, attacker.attack - defender.defense + self.rng.randint(0, 2))
        defender.take_damage(damage)
        p = self.player

        if defender is p:
            self.message(f"{attacker.name} бьёт вас: −{damage}", Color.BAD)
            if not p.alive:
                self.dead = True
                self.killer = attacker.name
                self.message(f"Вы погибли. Убийца: {attacker.name}.", Color.BAD)
            return

        assert isinstance(defender, Monster)
        if defender.alive:
            self.message(f"Вы бьёте: {defender.name} −{damage} (осталось {defender.hp})")
            return

        self.monsters.remove(defender)
        p.kills += 1
        self.message(f"Враг повержен: {defender.name} (+{defender.xp_reward} опыта)", Color.GOOD)
        for text in p.gain_xp(defender.xp_reward):
            self.message(text, Color.INFO)

    def _pick_up(self, item: Item) -> None:
        self.items.remove(item)
        p = self.player
        key = item.kind.key
        if key == 'potion':
            p.potions += 1
            text = "Вы подбираете зелье лечения. H — выпить."
        elif key == 'sword':
            p.attack += 1
            text = "Вы находите меч: атака +1."
        elif key == 'shield':
            p.defense += 1
            text = "Вы находите щит: защита +1."
        else:
            p.max_hp += 5
            p.heal(5)
            text = "Амулет жизни: максимум здоровья +5."
        self.message(text, Color.LOOT)

    def _drink(self) -> bool:
        p = self.player
        if p.potions == 0:
            self.message("Зелий нет.", Color.MUTED)
            return False
        if p.hp == p.max_hp:
            self.message("Вы и так полностью здоровы.", Color.MUTED)
            return False
        p.potions -= 1
        healed = p.heal(max(10, p.max_hp * 2 // 5))
        self.message(f"Вы выпиваете зелье: +{healed} здоровья.", Color.GOOD)
        return True

    def _descend(self) -> None:
        if not self.on_stairs:
            self.message("Лестницы здесь нет. Ищите «>».", Color.MUTED)
            return
        self.next_level()
        healed = self.player.heal(self.player.max_hp // 4)
        self.message(f"Глубина {self.depth}. Передышка на лестнице: +{healed} здоровья.", Color.INFO)

    # ── Ход монстров ─────────────────────────────────────────────────────

    def _end_turn(self) -> None:
        self.turn += 1
        p = self.player
        if self.turn % REGEN_EVERY == 0:
            p.heal(1)
        self._update_fov()

        dist = self.map.distance_map(p.pos, CHASE_RANGE)
        for monster in list(self.monsters):
            self._monster_turn(monster, dist)
            if self.dead:
                break

    def _monster_turn(self, m: Monster, dist: Dict[Point, int]) -> None:
        p = self.player
        if m.pos in self.map.visible:
            m.alert = MONSTER_MEMORY
        elif m.alert > 0:
            m.alert -= 1

        if abs(m.x - p.x) + abs(m.y - p.y) == 1:
            self._attack(m, p)
            return

        if m.alert > 0 and m.pos in dist:
            best, best_d = None, dist[m.pos]
            for dx, dy in DIRS4:
                nxt = (m.x + dx, m.y + dy)
                d = dist.get(nxt)
                if d is not None and d < best_d and not self.actor_at(*nxt):
                    best, best_d = nxt, d
            if best:
                m.x, m.y = best
            return

        if self.rng.random() < 0.3:                # не видит игрока — бродит
            dx, dy = self.rng.choice(DIRS4)
            nx, ny = m.x + dx, m.y + dy
            if self.map.is_walkable(nx, ny) and not self.actor_at(nx, ny):
                m.x, m.y = nx, ny

    def _update_fov(self) -> None:
        self.map.compute_fov(self.player.x, self.player.y, FOV_RADIUS)


# ═══════════════════════════════════════════════════════════════════════════
#  Отрисовка
# ═══════════════════════════════════════════════════════════════════════════

UI_FONTS = "segoeui,arial,dejavusans,liberationsans,freesans"
GLYPH_FONTS = "consolas,dejavusansmono,menlo,couriernew,liberationmono,monospace"


def _mix(color: RGB, base: RGB, k: float) -> RGB:
    """Смешивание цветов: k=1 — исходный цвет, k=0 — цвет фона."""
    return tuple(int(b + (c - b) * k) for c, b in zip(color, base))  # type: ignore[return-value]


class Renderer:
    def __init__(self, screen: pygame.Surface) -> None:
        self.screen = screen
        self.glyph_font = pygame.font.SysFont(GLYPH_FONTS, TILE - 2, bold=True)
        self.font = pygame.font.SysFont(UI_FONTS, 15)
        self.font_small = pygame.font.SysFont(UI_FONTS, 13)
        self.font_bold = pygame.font.SysFont(UI_FONTS, 15, bold=True)
        self.font_big = pygame.font.SysFont(UI_FONTS, 44, bold=True)
        self._glyphs: Dict[Tuple[str, RGB], pygame.Surface] = {}
        self._shade = pygame.Surface((SCREEN_W, MAP_H * TILE), pygame.SRCALPHA)
        self._shade.fill((0, 0, 0, 175))

    def draw(self, game: Game) -> None:
        self.screen.fill(Color.BG)
        self._draw_map(game.map)
        visible = game.map.visible
        for item in game.items:
            if item.pos in visible:
                self._blit_glyph(item.glyph, item.color, item.x, item.y)
        for monster in game.monsters:
            if monster.pos in visible:
                self._draw_monster(monster)
        p = game.player
        center = (p.x * TILE + TILE // 2, p.y * TILE + TILE // 2)
        pygame.draw.circle(self.screen, Color.PLAYER_GLOW, center, TILE // 2)
        self._blit_glyph(p.glyph, p.color, p.x, p.y)
        self._draw_panel(game)
        if game.dead:
            self._draw_death(game)

    # ── Карта и существа ─────────────────────────────────────────────────

    def _glyph(self, ch: str, color: RGB) -> pygame.Surface:
        key = (ch, color)
        if key not in self._glyphs:                # рендер текста дорогой — кешируем
            self._glyphs[key] = self.glyph_font.render(ch, True, color)
        return self._glyphs[key]

    def _blit_glyph(self, ch: str, color: RGB, x: int, y: int) -> None:
        surface = self._glyph(ch, color)
        rect = surface.get_rect(center=(x * TILE + TILE // 2, y * TILE + TILE // 2))
        self.screen.blit(surface, rect)

    def _draw_map(self, game_map: GameMap) -> None:
        for x, y in game_map.explored:
            tile = game_map.tiles[y][x]
            lit = (x, y) in game_map.visible
            rect = pygame.Rect(x * TILE, y * TILE, TILE, TILE)
            if tile is Tile.WALL:
                pygame.draw.rect(self.screen, Color.WALL_LIT if lit else Color.WALL_DARK, rect)
                if lit:
                    pygame.draw.rect(self.screen, Color.WALL_LIT_EDGE, rect, 1)
                continue
            pygame.draw.rect(self.screen, Color.FLOOR_LIT if lit else Color.FLOOR_DARK, rect)
            if tile is Tile.STAIRS:
                self._blit_glyph('>', Color.STAIRS if lit else Color.STAIRS_DARK, x, y)
            elif lit:
                pygame.draw.rect(self.screen, Color.FLOOR_DOT, (rect.centerx - 1, rect.centery - 1, 2, 2))

    def _draw_monster(self, m: Monster) -> None:
        self._blit_glyph(m.glyph, m.color, m.x, m.y)
        if m.hp < m.max_hp:
            width = TILE - 4
            x, y = m.x * TILE + 2, m.y * TILE + 1
            pygame.draw.rect(self.screen, Color.HP_BG, (x, y, width, 2))
            pygame.draw.rect(self.screen, Color.HP, (x, y, max(1, width * m.hp // m.max_hp), 2))

    # ── Панель ───────────────────────────────────────────────────────────

    def _text(self, text: str, font: pygame.font.Font, color: RGB, x: int, y: int) -> None:
        self.screen.blit(font.render(text, True, color), (x, y))

    def _draw_panel(self, game: Game) -> None:
        top = MAP_H * TILE
        pygame.draw.rect(self.screen, Color.PANEL, (0, top, SCREEN_W, PANEL_H))
        pygame.draw.line(self.screen, Color.PANEL_LINE, (0, top), (SCREEN_W, top), 2)

        p = game.player
        x0, bar_w = 16, 270

        y = top + 12
        pygame.draw.rect(self.screen, Color.HP_BG, (x0, y, bar_w, 20), border_radius=4)
        if p.hp:
            pygame.draw.rect(self.screen, Color.HP, (x0, y, bar_w * p.hp // p.max_hp, 20), border_radius=4)
        label = self.font_bold.render(f"Здоровье  {p.hp} / {p.max_hp}", True, Color.TEXT)
        self.screen.blit(label, label.get_rect(center=(x0 + bar_w // 2, y + 10)))

        y += 26
        pygame.draw.rect(self.screen, Color.XP_BG, (x0, y, bar_w, 6), border_radius=3)
        pygame.draw.rect(self.screen, Color.XP, (x0, y, bar_w * p.xp // p.xp_to_next, 6), border_radius=3)

        y += 14
        stats = (
            f"Глубина {game.depth}   ·   рекорд {game.best_depth}",
            f"Уровень {p.level}   ·   опыт {p.xp} / {p.xp_to_next}",
            f"Атака {p.attack}   ·   Защита {p.defense}   ·   Зелья {p.potions}",
        )
        for line in stats:
            self._text(line, self.font, Color.TEXT, x0, y)
            y += 20

        pygame.draw.line(self.screen, Color.PANEL_LINE, (304, top + 12), (304, top + 116))
        log_x, log_y = 320, top + 12
        messages = list(game.log)
        for k, (text, color) in enumerate(messages):
            age = len(messages) - 1 - k              # старые сообщения бледнее
            self._text(text, self.font, _mix(color, Color.PANEL, 1 - age * 0.13), log_x, log_y)
            log_y += 18

        hint = ("WASD / стрелки — ход и атака    H — зелье    Enter — вниз по лестнице    "
                "Пробел — ждать    Esc — выход")
        self._text(hint, self.font_small, Color.MUTED, x0, top + PANEL_H - 22)

    def _draw_death(self, game: Game) -> None:
        self.screen.blit(self._shade, (0, 0))
        cx, cy = SCREEN_W // 2, MAP_H * TILE // 2
        p = game.player
        lines = [
            (self.font_big, "Вы погибли", Color.BAD, -60),
            (self.font, f"Убийца: {game.killer}", Color.TEXT, 0),
            (self.font, f"Глубина {game.depth}  ·  Уровень {p.level}  ·  Врагов повержено: {p.kills}",
             Color.TEXT, 26),
            (self.font_bold, "R — начать заново     Esc — выход", Color.LOOT, 70),
        ]
        for font, text, color, offset in lines:
            surface = font.render(text, True, color)
            self.screen.blit(surface, surface.get_rect(center=(cx, cy + offset)))


# ═══════════════════════════════════════════════════════════════════════════
#  Ввод и главный цикл
# ═══════════════════════════════════════════════════════════════════════════

MOVE_KEYS = {
    pygame.K_UP: (0, -1), pygame.K_w: (0, -1), pygame.K_KP8: (0, -1),
    pygame.K_DOWN: (0, 1), pygame.K_s: (0, 1), pygame.K_KP2: (0, 1),
    pygame.K_LEFT: (-1, 0), pygame.K_a: (-1, 0), pygame.K_KP4: (-1, 0),
    pygame.K_RIGHT: (1, 0), pygame.K_d: (1, 0), pygame.K_KP6: (1, 0),
}

# Физические клавиши: управление работает и на русской раскладке
SCAN_MOVE = {
    getattr(pygame, 'KSCAN_W', -1): (0, -1),
    getattr(pygame, 'KSCAN_S', -2): (0, 1),
    getattr(pygame, 'KSCAN_A', -3): (-1, 0),
    getattr(pygame, 'KSCAN_D', -4): (1, 0),
}
SCAN_H = getattr(pygame, 'KSCAN_H', -5)
SCAN_R = getattr(pygame, 'KSCAN_R', -6)


def event_to_action(event: pygame.event.Event) -> Optional[Action]:
    scancode = getattr(event, 'scancode', None)

    if event.key in MOVE_KEYS:
        return ('move', *MOVE_KEYS[event.key])
    if scancode in SCAN_MOVE:
        return ('move', *SCAN_MOVE[scancode])
    if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER) or event.unicode == '>':
        return ('descend',)
    if event.key in (pygame.K_SPACE, pygame.K_PERIOD, pygame.K_KP5):
        return ('wait',)
    if event.key == pygame.K_h or scancode == SCAN_H:
        return ('drink',)
    if event.key == pygame.K_r or scancode == SCAN_R:
        return ('restart',)
    return None


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Пошаговый roguelike на Pygame")
    parser.add_argument('--seed', type=int, help="зерно генерации: одинаковое подземелье при каждом запуске")
    args = parser.parse_args(argv)

    pygame.init()
    pygame.display.set_caption("Roguelike — подземелье")
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.key.set_repeat(180, 80)                 # зажатая клавиша — серия шагов

    game = Game(args.seed)
    renderer = Renderer(screen)
    clock = pygame.time.Clock()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                    continue
                action = event_to_action(event)
                if action:
                    game.perform(action)
        renderer.draw(game)
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    return 0


if __name__ == '__main__':
    sys.exit(main())
