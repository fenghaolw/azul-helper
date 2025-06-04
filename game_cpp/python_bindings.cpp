#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "tile.h"
#include "action.h"
#include "player_board.h"
#include "factory.h"
#include "game_state.h"

#ifdef WITH_OPENSPIEL
#include "azul_openspiel.h"
#include "mcts_agent.h"
#endif

namespace py = pybind11;

PYBIND11_MODULE(azul_cpp_bindings, m) {
    m.doc() = "Azul game implementation in C++";
    
    // TileColor enum
    py::enum_<azul::TileColor>(m, "TileColor")
        .value("BLUE", azul::TileColor::BLUE)
        .value("YELLOW", azul::TileColor::YELLOW)
        .value("RED", azul::TileColor::RED)
        .value("BLACK", azul::TileColor::BLACK)
        .value("WHITE", azul::TileColor::WHITE)
        .value("FIRST_PLAYER", azul::TileColor::FIRST_PLAYER);
    
    // Tile class
    py::class_<azul::Tile>(m, "Tile")
        .def(py::init<azul::TileColor>())
        .def("color", &azul::Tile::color)
        .def("is_first_player_marker", &azul::Tile::is_first_player_marker)
        .def("to_string", &azul::Tile::to_string)
        .def("__eq__", &azul::Tile::operator==)
        .def("__repr__", &azul::Tile::to_string)
        .def_static("get_tile", &azul::Tile::get_tile, py::return_value_policy::reference)
        .def_static("create_standard_tiles", &azul::Tile::create_standard_tiles)
        .def_static("create_first_player_marker", &azul::Tile::create_first_player_marker, 
                   py::return_value_policy::reference);
    
    // Action class
    py::class_<azul::Action>(m, "Action")
        .def(py::init<int, azul::TileColor, int>())
        .def("source", &azul::Action::source)
        .def("color", &azul::Action::color)
        .def("destination", &azul::Action::destination)
        .def("to_string", &azul::Action::to_string)
        .def("__eq__", &azul::Action::operator==)
        .def("__repr__", &azul::Action::to_string)
        .def("__hash__", &azul::Action::hash);
    
    // PatternLine class
    py::class_<azul::PatternLine>(m, "PatternLine")
        .def(py::init<int>())
        .def("capacity", &azul::PatternLine::capacity)
        .def("tiles", &azul::PatternLine::tiles, py::return_value_policy::reference_internal)
        .def("has_color", &azul::PatternLine::has_color)
        .def("color", &azul::PatternLine::color)
        .def("can_add_tiles", &azul::PatternLine::can_add_tiles)
        .def("add_tiles", &azul::PatternLine::add_tiles)
        .def("is_complete", &azul::PatternLine::is_complete)
        .def("clear", &azul::PatternLine::clear)
        .def("get_wall_tile", &azul::PatternLine::get_wall_tile)
        .def("copy", &azul::PatternLine::copy);
    
    // Wall class
    py::class_<azul::Wall>(m, "Wall")
        .def(py::init<>())
        .def("can_place_tile", &azul::Wall::can_place_tile)
        .def("place_tile", &azul::Wall::place_tile)
        .def("is_row_complete", &azul::Wall::is_row_complete)
        .def("is_column_complete", &azul::Wall::is_column_complete)
        .def("is_color_complete", &azul::Wall::is_color_complete)
        .def("get_completed_rows", &azul::Wall::get_completed_rows)
        .def("get_completed_columns", &azul::Wall::get_completed_columns)
        .def("get_completed_colors", &azul::Wall::get_completed_colors)
        .def("copy", &azul::Wall::copy);
    
    // PlayerBoard class
    py::class_<azul::PlayerBoard>(m, "PlayerBoard")
        .def(py::init<>())
        .def("can_place_tiles_on_pattern_line", &azul::PlayerBoard::can_place_tiles_on_pattern_line)
        .def("place_tiles_on_pattern_line", &azul::PlayerBoard::place_tiles_on_pattern_line)
        .def("place_tiles_on_floor_line", &azul::PlayerBoard::place_tiles_on_floor_line)
        .def("end_round_scoring", &azul::PlayerBoard::end_round_scoring)
        .def("final_scoring", &azul::PlayerBoard::final_scoring)
        .def("has_first_player_marker", &azul::PlayerBoard::has_first_player_marker)
        .def("remove_first_player_marker", &azul::PlayerBoard::remove_first_player_marker)
        .def("pattern_lines", &azul::PlayerBoard::pattern_lines, py::return_value_policy::reference_internal)
        .def("wall", &azul::PlayerBoard::wall, py::return_value_policy::reference_internal)
        .def("floor_line", &azul::PlayerBoard::floor_line, py::return_value_policy::reference_internal)
        .def("score", &azul::PlayerBoard::score)
        .def("copy", &azul::PlayerBoard::copy);
    
    // Factory class
    py::class_<azul::Factory>(m, "Factory")
        .def(py::init<>())
        .def("fill_from_bag", &azul::Factory::fill_from_bag)
        .def("take_tiles", &azul::Factory::take_tiles)
        .def("is_empty", &azul::Factory::is_empty)
        .def("has_color", &azul::Factory::has_color)
        .def("get_available_colors", &azul::Factory::get_available_colors)
        .def("tiles", &azul::Factory::tiles, py::return_value_policy::reference_internal)
        .def("copy", &azul::Factory::copy);
    
    // CenterArea class
    py::class_<azul::CenterArea>(m, "CenterArea")
        .def(py::init<>())
        .def("add_tiles", &azul::CenterArea::add_tiles)
        .def("add_first_player_marker", &azul::CenterArea::add_first_player_marker)
        .def("take_tiles", &azul::CenterArea::take_tiles)
        .def("is_empty", &azul::CenterArea::is_empty)
        .def("has_color", &azul::CenterArea::has_color)
        .def("get_available_colors", &azul::CenterArea::get_available_colors)
        .def("clear", &azul::CenterArea::clear)
        .def("tiles", &azul::CenterArea::tiles, py::return_value_policy::reference_internal)
        .def("has_first_player_marker", &azul::CenterArea::has_first_player_marker)
        .def("copy", &azul::CenterArea::copy);
    
    // FactoryArea class
    py::class_<azul::FactoryArea>(m, "FactoryArea")
        .def(py::init<int>())
        .def("setup_round", &azul::FactoryArea::setup_round)
        .def("take_from_factory", &azul::FactoryArea::take_from_factory)
        .def("take_from_center", &azul::FactoryArea::take_from_center)
        .def("is_round_over", &azul::FactoryArea::is_round_over)
        .def("get_available_moves", &azul::FactoryArea::get_available_moves)
        .def("num_factories", &azul::FactoryArea::num_factories)
        .def("factories", &azul::FactoryArea::factories, py::return_value_policy::reference_internal)
        .def("center", &azul::FactoryArea::center, py::return_value_policy::reference_internal)
        .def("copy", &azul::FactoryArea::copy);
    
    // GameState class
    py::class_<azul::GameState>(m, "GameState")
        .def(py::init<int, int>(), py::arg("num_players") = 2, py::arg("seed") = -1)
        .def("get_legal_actions", &azul::GameState::get_legal_actions, py::arg("player_id") = -1)
        .def("is_action_legal", &azul::GameState::is_action_legal, py::arg("action"), py::arg("player_id") = -1)
        .def("apply_action", &azul::GameState::apply_action, py::arg("action"), py::arg("skip_validation") = false)
        .def("is_game_over", &azul::GameState::is_game_over)
        .def("get_winner", &azul::GameState::get_winner)
        .def("get_scores", &azul::GameState::get_scores)
        .def("get_state_vector", &azul::GameState::get_state_vector)
        .def("current_player", &azul::GameState::current_player)
        .def("num_players", &azul::GameState::num_players)
        .def("players", &azul::GameState::players, py::return_value_policy::reference_internal)
        .def("factory_area", &azul::GameState::factory_area, py::return_value_policy::reference_internal)
        .def("bag", &azul::GameState::bag, py::return_value_policy::reference_internal)
        .def("discard_pile", &azul::GameState::discard_pile, py::return_value_policy::reference_internal)
        .def("round_number", &azul::GameState::round_number)
        .def("copy", &azul::GameState::copy);
    
    // Factory function
    m.def("create_game", &azul::create_game, py::arg("num_players") = 2, py::arg("seed") = -1);

#ifdef WITH_OPENSPIEL
    // OpenSpiel integration
    py::class_<azul::AzulGame>(m, "AzulGame")
        .def(py::init<const open_spiel::GameParameters&>())
        .def("NumDistinctActions", &azul::AzulGame::NumDistinctActions)
        .def("NewInitialState", &azul::AzulGame::NewInitialState)
        .def("NumPlayers", &azul::AzulGame::NumPlayers)
        .def("MinUtility", &azul::AzulGame::MinUtility)
        .def("MaxUtility", &azul::AzulGame::MaxUtility)
        .def("ObservationTensorShape", &azul::AzulGame::ObservationTensorShape)
        .def("MaxGameLength", &azul::AzulGame::MaxGameLength);
    
    py::class_<azul::AzulState>(m, "AzulState")
        .def("CurrentPlayer", &azul::AzulState::CurrentPlayer)
        .def("LegalActions", &azul::AzulState::LegalActions)
        .def("ActionToString", &azul::AzulState::ActionToString)
        .def("ToString", &azul::AzulState::ToString)
        .def("IsTerminal", &azul::AzulState::IsTerminal)
        .def("Returns", &azul::AzulState::Returns)
        .def("ObservationString", &azul::AzulState::ObservationString)
        .def("Clone", &azul::AzulState::Clone);
    
    // MCTS Agent
    py::class_<azul::AzulMCTSAgent>(m, "AzulMCTSAgent")
        .def(py::init<int, int, double, int>(), 
             py::arg("player_id"), py::arg("num_simulations") = 1000, 
             py::arg("uct_c") = 1.4, py::arg("seed") = -1)
        .def("get_action", &azul::AzulMCTSAgent::get_action)
        .def("get_action_probabilities", &azul::AzulMCTSAgent::get_action_probabilities, 
             py::arg("state"), py::arg("temperature") = 1.0)
        .def("reset", &azul::AzulMCTSAgent::reset)
        .def("set_num_simulations", &azul::AzulMCTSAgent::set_num_simulations)
        .def("set_uct_c", &azul::AzulMCTSAgent::set_uct_c)
        .def("player_id", &azul::AzulMCTSAgent::player_id);
    
    m.def("create_mcts_agent", &azul::create_mcts_agent, 
          py::arg("player_id"), py::arg("num_simulations") = 1000, 
          py::arg("uct_c") = 1.4, py::arg("seed") = -1);
#endif
} 