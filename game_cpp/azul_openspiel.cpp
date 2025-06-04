#ifdef WITH_OPENSPIEL

#include "azul_openspiel.h"
#include "open_spiel/spiel_utils.h"

namespace azul {

// Color mapping for action encoding
const std::unordered_map<TileColor, int> AzulState::COLOR_TO_IDX = {
    {TileColor::BLUE, 0},
    {TileColor::YELLOW, 1},
    {TileColor::RED, 2},
    {TileColor::BLACK, 3},
    {TileColor::WHITE, 4}
};

const std::unordered_map<int, TileColor> AzulState::IDX_TO_COLOR = {
    {0, TileColor::BLUE},
    {1, TileColor::YELLOW},
    {2, TileColor::RED},
    {3, TileColor::BLACK},
    {4, TileColor::WHITE}
};

// AzulGame implementation
AzulGame::AzulGame(const open_spiel::GameParameters& params) 
    : Game(open_spiel::GameType{
        /*short_name=*/"azul",
        /*long_name=*/"Azul",
        open_spiel::GameType::Dynamics::kSequential,
        open_spiel::GameType::ChanceMode::kDeterministic,
        open_spiel::GameType::Information::kPerfectInformation,
        open_spiel::GameType::Utility::kZeroSum,
        open_spiel::GameType::RewardModel::kTerminal,
        /*max_num_players=*/4,
        /*min_num_players=*/2,
        /*provides_information_state_string=*/true,
        /*provides_information_state_tensor=*/true,
        /*provides_observation_string=*/true,
        /*provides_observation_tensor=*/true,
        /*parameter_specification=*/{}
    }, open_spiel::GameInfo{
        /*num_distinct_actions=*/180,
        /*max_chance_outcomes=*/0,
        /*num_players=*/open_spiel::ParameterValue<int>(params, "players", 2),
        /*min_utility=*/-100.0,
        /*max_utility=*/100.0,
        /*utility_sum=*/0.0,
        /*max_game_length=*/200
    }, params) {
    
    num_players_ = open_spiel::ParameterValue<int>(params, "players", 2);
    seed_ = open_spiel::ParameterValue<int>(params, "seed", -1);
}

std::unique_ptr<open_spiel::State> AzulGame::NewInitialState() const {
    return std::make_unique<AzulState>(shared_from_this(), num_players_, seed_);
}

std::vector<int> AzulGame::ObservationTensorShape() const {
    return {16, 16, 4}; // 1024 total elements
}

// AzulState implementation
AzulState::AzulState(std::shared_ptr<const open_spiel::Game> game, int num_players, int seed)
    : State(game), game_state_(num_players, seed) {}

open_spiel::Player AzulState::CurrentPlayer() const {
    if (game_state_.is_game_over()) {
        return open_spiel::kTerminalPlayerId;
    }
    return game_state_.current_player();
}

std::vector<open_spiel::Action> AzulState::LegalActions() const {
    if (game_state_.is_game_over()) {
        return {};
    }
    
    auto azul_actions = game_state_.get_legal_actions();
    std::vector<open_spiel::Action> actions;
    actions.reserve(azul_actions.size());
    
    for (const auto& action : azul_actions) {
        actions.push_back(azul_action_to_int(action));
    }
    
    return actions;
}

Action AzulState::int_to_azul_action(int action_int) const {
    // Decode: source_idx * 30 + color_idx * 6 + dest_idx
    int source_idx = action_int / 30;
    int remainder = action_int % 30;
    int color_idx = remainder / 6;
    int dest_idx = remainder % 6;
    
    int source = source_idx - 1; // 0 becomes -1, 1-5 becomes 0-4
    TileColor color = IDX_TO_COLOR.at(color_idx);
    int destination = dest_idx - 1; // 0 becomes -1, 1-5 becomes 0-4
    
    return Action(source, color, destination);
}

int AzulState::azul_action_to_int(const Action& action) const {
    int source_idx = action.source() + 1; // -1 becomes 0, 0-4 becomes 1-5
    int dest_idx = action.destination() + 1; // -1 becomes 0, 0-4 becomes 1-5
    int color_idx = COLOR_TO_IDX.at(action.color());
    
    return source_idx * 30 + color_idx * 6 + dest_idx;
}

std::string AzulState::ActionToString(open_spiel::Player player, open_spiel::Action action_id) const {
    Action action = int_to_azul_action(action_id);
    return action.to_string();
}

std::string AzulState::ToString() const {
    return "Azul game state"; // Could be more detailed
}

bool AzulState::IsTerminal() const {
    return game_state_.is_game_over();
}

std::vector<double> AzulState::Returns() const {
    if (!game_state_.is_game_over()) {
        return std::vector<double>(game_state_.num_players(), 0.0);
    }
    
    auto scores = game_state_.get_scores();
    std::vector<double> returns(scores.size());
    
    // Convert to utilities (winner gets +1, others get -1 for 2-player, proportional for multi-player)
    if (game_state_.num_players() == 2) {
        int winner = game_state_.get_winner();
        returns[winner] = 1.0;
        returns[1 - winner] = -1.0;
    } else {
        // For multi-player, normalize scores
        double max_score = *std::max_element(scores.begin(), scores.end());
        double min_score = *std::min_element(scores.begin(), scores.end());
        double range = max_score - min_score;
        
        if (range > 0) {
            for (size_t i = 0; i < scores.size(); ++i) {
                returns[i] = 2.0 * (scores[i] - min_score) / range - 1.0; // Scale to [-1, 1]
            }
        }
    }
    
    return returns;
}

std::string AzulState::ObservationString(open_spiel::Player player) const {
    return ToString(); // Could be more detailed
}

void AzulState::ObservationTensor(open_spiel::Player player, absl::Span<float> values) const {
    auto state_vector = game_state_.get_state_vector();
    
    // Pad or truncate to fit the observation tensor shape
    size_t tensor_size = 16 * 16 * 4; // 1024
    for (size_t i = 0; i < tensor_size; ++i) {
        if (i < state_vector.size()) {
            values[i] = state_vector[i];
        } else {
            values[i] = 0.0f;
        }
    }
}

std::unique_ptr<open_spiel::State> AzulState::Clone() const {
    auto cloned_state = std::make_unique<AzulState>(game_, 0); // Dummy initialization
    cloned_state->game_state_ = game_state_.copy();
    return cloned_state;
}

void AzulState::DoApplyAction(open_spiel::Action action) {
    Action azul_action = int_to_azul_action(action);
    game_state_.apply_action(azul_action, true); // Skip validation for performance
}

} // namespace azul

#endif // WITH_OPENSPIEL 