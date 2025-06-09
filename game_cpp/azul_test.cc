#include "azul.h"

#include <cmath>

#include "gtest/gtest.h"

namespace open_spiel::azul {
namespace {

TEST(AzulTest, ReturnsFunction) {
  auto game = std::make_shared<const AzulGame>(GameParameters{});
  auto state = std::make_unique<AzulState>(game);
  const double scaling_factor = 40.0;

  // Test case 1: Player 0 wins
  // Player 0: 50 points
  // Player 1: 30 points
  state->player_boards_[0].score = 50;
  state->player_boards_[1].score = 30;
  state->game_ended_ = true;

  std::vector<double> returns = state->Returns();

  // Calculate expected rewards
  double score_diff = 50.0 - 30.0;
  double expected_reward = std::tanh(score_diff / scaling_factor);

  // Verify exact reward values
  EXPECT_NEAR(returns[0], expected_reward, 1e-6);
  EXPECT_NEAR(returns[1], -expected_reward, 1e-6);

  // Test case 2: Player 1 wins
  // Player 0: 30 points
  // Player 1: 50 points
  state->player_boards_[0].score = 30;
  state->player_boards_[1].score = 50;

  returns = state->Returns();

  // Calculate expected rewards
  score_diff = 30.0 - 50.0;
  expected_reward = std::tanh(score_diff / scaling_factor);

  // Verify exact reward values
  EXPECT_NEAR(returns[0], expected_reward, 1e-6);
  EXPECT_NEAR(returns[1], -expected_reward, 1e-6);

  // Test case 3: Tie
  // Player 0: 40 points
  // Player 1: 40 points
  state->player_boards_[0].score = 40;
  state->player_boards_[1].score = 40;

  returns = state->Returns();

  // For a tie, score_diff is 0, so tanh(0) = 0
  EXPECT_NEAR(returns[0], 0.0, 1e-6);
  EXPECT_NEAR(returns[1], 0.0, 1e-6);

  // Test case 4: Large score difference
  // Player 0: 100 points
  // Player 1: 0 points
  state->player_boards_[0].score = 100;
  state->player_boards_[1].score = 0;

  returns = state->Returns();

  // Calculate expected rewards
  score_diff = 100.0 - 0.0;
  expected_reward = std::tanh(score_diff / scaling_factor);

  // Verify exact reward values
  EXPECT_NEAR(returns[0], expected_reward, 1e-6);
  EXPECT_NEAR(returns[1], -expected_reward, 1e-6);

  // Verify that rewards are bounded by tanh
  EXPECT_LT(returns[0], 1.0);
  EXPECT_GT(returns[0], 0.0);
  EXPECT_GT(returns[1], -1.0);
  EXPECT_LT(returns[1], 0.0);
}

}  // namespace
}  // namespace open_spiel::azul