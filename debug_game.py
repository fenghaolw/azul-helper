#!/usr/bin/env python3

from azul_rl.game.game_state import create_game

def debug_game_simulation():
    game = create_game(num_players=2, seed=42)
    
    # Just analyze the first action in detail
    actions = game.get_legal_actions()
    print(f'Total legal actions: {len(actions)}')
    
    # Group actions by destination
    pattern_actions = [a for a in actions if a.destination >= 0]
    floor_actions = [a for a in actions if a.destination == -1]
    
    print(f'Pattern line actions: {len(pattern_actions)}')
    print(f'Floor line actions: {len(floor_actions)}')
    
    if pattern_actions:
        print('Pattern line actions:')
        for action in pattern_actions[:5]:  # Show first 5
            print(f'  {action}')
    
    # Check why pattern line actions might not be available
    player = game.players[game.current_player]
    print(f'\nPlayer {game.current_player} state:')
    
    # Check pattern lines
    for i, line in enumerate(player.pattern_lines):
        print(f'  Pattern line {i}: {len(line.tiles)}/{line.capacity} tiles, color={line.color}')
    
    # Check wall
    print('  Wall state:')
    for row in range(5):
        filled_positions = [col for col in range(5) if player.wall.filled[row][col]]
        if filled_positions:
            print(f'    Row {row}: filled positions {filled_positions}')
    
    # Check available tiles in factories and center
    print('\nAvailable tiles:')
    for i, factory in enumerate(game.factory_area.factories):
        if factory.tiles:
            colors = {}
            for tile in factory.tiles:
                colors[tile.color] = colors.get(tile.color, 0) + 1
            print(f'  Factory {i}: {colors}')
    
    if game.factory_area.center.tiles:
        colors = {}
        for tile in game.factory_area.center.tiles:
            colors[tile.color] = colors.get(tile.color, 0) + 1
        print(f'  Center: {colors}')
    
    # Test specific placement
    if game.factory_area.factories[0].tiles:
        test_tiles = [game.factory_area.factories[0].tiles[0]]
        print(f'\nTesting placement of {test_tiles[0].color} on pattern lines:')
        for i in range(5):
            can_place = player.can_place_tiles_on_pattern_line(i, test_tiles)
            wall_can_place = player.wall.can_place_tile(i, test_tiles[0].color)
            line_can_add = player.pattern_lines[i].can_add_tiles(test_tiles)
            print(f'  Line {i}: can_place={can_place}, wall_can_place={wall_can_place}, line_can_add={line_can_add}')

if __name__ == "__main__":
    debug_game_simulation() 