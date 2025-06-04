#pragma once

#ifdef WITH_OPENSPIEL

#include "game_state.h"
#include "open_spiel/spiel.h"
#include <memory>

namespace azul {

class AzulGame : public open_spiel::Game {
public:
    explicit AzulGame(const open_spiel::GameParameters& params);
    
    [[nodiscard]] auto NumDistinctActions() const -> int override { return 180; }
    [[nodiscard]] auto NewInitialState() const -> std::unique_ptr<open_spiel::State> override;
    [[nodiscard]] auto NumPlayers() const -> int override { return num_players_; }
    [[nodiscard]] auto MinUtility() const -> double override { return -100.0; }
    [[nodiscard]] auto MaxUtility() const -> double override { return 100.0; }
    [[nodiscard]] auto ObservationTensorShape() const -> std::vector<int> override;
    [[nodiscard]] auto MaxGameLength() const -> int override { return 200; }

private:
    int num_players_;
    int seed_;
};

class AzulState : public open_spiel::State {
public:
    AzulState(std::shared_ptr<const open_spiel::Game> game, int num_players, int seed = -1);
    
    [[nodiscard]] auto CurrentPlayer() const -> open_spiel::Player override;
    [[nodiscard]] auto LegalActions() const -> std::vector<open_spiel::Action> override;
    [[nodiscard]] auto ActionToString(open_spiel::Player player, open_spiel::Action action_id) const -> std::string override;
    [[nodiscard]] auto ToString() const -> std::string override;
    [[nodiscard]] auto IsTerminal() const -> bool override;
    [[nodiscard]] auto Returns() const -> std::vector<double> override;
    [[nodiscard]] auto ObservationString(open_spiel::Player player) const -> std::string override;
    void ObservationTensor(open_spiel::Player player, absl::Span<float> values) const override;
    [[nodiscard]] auto Clone() const -> std::unique_ptr<open_spiel::State> override;
    void DoApplyAction(open_spiel::Action action) override;

private:
    GameState game_state_;
    
    // Action conversion helpers
    [[nodiscard]] auto int_to_azul_action(int action_int) const -> Action;
    [[nodiscard]] auto azul_action_to_int(const Action& action) const -> int;
    
    // Color mapping for action encoding
    static const std::unordered_map<TileColor, int> COLOR_TO_IDX;
    static const std::unordered_map<int, TileColor> IDX_TO_COLOR;
};

} // namespace azul

#endif // WITH_OPENSPIEL 