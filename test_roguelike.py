"""Тесты логики roguelike — окно pygame не открывается.  Запуск:  python -m unittest -v"""

import random
import unittest

from roguelike import (
    DIRS4, MAP_H, MAP_W, Game, GameMap, Monster, MONSTER_KINDS, Player,
    Rect, Tile, bresenham, generate_dungeon,
)


class TestGeometry(unittest.TestCase):
    def test_bresenham_endpoints_and_continuity(self):
        line = list(bresenham(0, 0, 7, 3))
        self.assertEqual(line[0], (0, 0))
        self.assertEqual(line[-1], (7, 3))
        for (x1, y1), (x2, y2) in zip(line, line[1:]):
            self.assertLessEqual(max(abs(x2 - x1), abs(y2 - y1)), 1)

    def test_rooms_with_gap_do_not_intersect(self):
        a = Rect(1, 1, 5, 5)                        # занимает x 1..5
        self.assertFalse(a.intersects(Rect(7, 1, 4, 4)))   # между ними стена x=6
        self.assertTrue(a.intersects(Rect(6, 1, 4, 4)))    # вплотную — нельзя


class TestDungeon(unittest.TestCase):
    def test_every_floor_tile_is_reachable(self):
        for seed in range(25):
            game_map, rooms = generate_dungeon(random.Random(seed))
            reachable = game_map.distance_map(rooms[0].center, max_dist=10_000)
            floor = {(x, y) for y in range(MAP_H) for x in range(MAP_W) if game_map.is_walkable(x, y)}
            self.assertEqual(floor, set(reachable), f"seed={seed}: есть недостижимые клетки")

    def test_stairs_exist_and_borders_are_walls(self):
        game_map, rooms = generate_dungeon(random.Random(1))
        sx, sy = rooms[-1].center
        self.assertIs(game_map.tiles[sy][sx], Tile.STAIRS)
        for x in range(MAP_W):
            self.assertIs(game_map.tiles[0][x], Tile.WALL)
            self.assertIs(game_map.tiles[MAP_H - 1][x], Tile.WALL)


class TestFieldOfView(unittest.TestCase):
    def make_open_map(self) -> GameMap:
        game_map = GameMap(21, 21)
        for y in range(1, 20):
            for x in range(1, 20):
                game_map.tiles[y][x] = Tile.FLOOR
        return game_map

    def test_wall_blocks_sight(self):
        game_map = self.make_open_map()
        for y in range(1, 20):
            game_map.tiles[y][12] = Tile.WALL       # сплошная стена справа
        game_map.compute_fov(10, 10, radius=8)
        self.assertIn((11, 10), game_map.visible)
        self.assertIn((12, 10), game_map.visible)    # саму стену видно
        self.assertNotIn((14, 10), game_map.visible) # а за ней — нет

    def test_radius_limits_sight(self):
        game_map = self.make_open_map()
        game_map.compute_fov(10, 10, radius=3)
        self.assertIn((13, 10), game_map.visible)
        self.assertNotIn((16, 10), game_map.visible)

    def test_explored_accumulates(self):
        game_map = self.make_open_map()
        game_map.compute_fov(5, 5, radius=3)
        game_map.compute_fov(15, 15, radius=3)
        self.assertIn((5, 5), game_map.explored)
        self.assertNotIn((5, 5), game_map.visible)


class TestCombatAndProgression(unittest.TestCase):
    def test_damage_is_at_least_one(self):
        game = Game(seed=0)
        tank = Monster(MONSTER_KINDS[0], 0, 0, depth=1)
        tank.defense = 999
        game._attack(game.player, tank)
        self.assertEqual(tank.hp, tank.max_hp - 1)

    def test_level_up(self):
        player = Player()
        player.hp = 1
        messages = player.gain_xp(player.xp_to_next)
        self.assertEqual(player.level, 2)
        self.assertEqual(player.hp, player.max_hp)
        self.assertEqual(len(messages), 1)

    def test_monsters_get_stronger_deeper(self):
        kind = MONSTER_KINDS[0]
        self.assertGreater(Monster(kind, 0, 0, depth=8).max_hp, Monster(kind, 0, 0, depth=1).max_hp)

    def test_first_room_is_safe(self):
        for seed in range(10):
            game = Game(seed=seed)
            px, py = game.player.pos
            for m in game.monsters:
                self.assertGreater(abs(m.x - px) + abs(m.y - py), 1)

    def test_walking_into_wall_does_not_spend_turn(self):
        game = Game(seed=3)
        game.monsters.clear()
        spot = next((x, y, dx, dy)
                    for y in range(MAP_H) for x in range(MAP_W) for dx, dy in DIRS4
                    if game.map.tiles[y][x] is Tile.FLOOR and not game.map.is_walkable(x + dx, y + dy))
        x, y, dx, dy = spot
        game.player.x, game.player.y = x, y
        turn = game.turn
        game.perform(('move', dx, dy))
        self.assertEqual(game.turn, turn)
        self.assertEqual(game.player.pos, (x, y))

    def test_descend_only_on_stairs(self):
        game = Game(seed=4)
        game.perform(('descend',))
        self.assertEqual(game.depth, 1)
        for y in range(MAP_H):
            for x in range(MAP_W):
                if game.map.tiles[y][x] is Tile.STAIRS:
                    game.player.x, game.player.y = x, y
        game.perform(('descend',))
        self.assertEqual(game.depth, 2)


class TestMonsterAI(unittest.TestCase):
    def test_alerted_monster_approaches_player(self):
        game = Game(seed=5)
        game.monsters.clear()
        game_map = game.map
        p = game.player
        dist = game_map.distance_map(p.pos, 30)
        far = next(pos for pos, d in dist.items() if d == 4)
        monster = Monster(MONSTER_KINDS[0], *far, depth=1)
        monster.alert = 5
        game.monsters.append(monster)
        game._monster_turn(monster, dist)
        self.assertEqual(dist[monster.pos], 3)


class TestSimulation(unittest.TestCase):
    def test_thousands_of_random_actions_do_not_crash(self):
        rng = random.Random(42)
        game = Game(seed=42)
        actions = [('move', 0, -1), ('move', 0, 1), ('move', -1, 0), ('move', 1, 0),
                   ('wait',), ('drink',), ('descend',)]
        deaths = 0
        for _ in range(5000):
            if game.dead:
                deaths += 1
                game.perform(('restart',))
            elif game.on_stairs:
                game.perform(('descend',))
            else:
                game.perform(rng.choice(actions))
            self.assertGreaterEqual(game.player.hp, 0)
            self.assertLessEqual(game.player.hp, game.player.max_hp)


if __name__ == '__main__':
    unittest.main()
