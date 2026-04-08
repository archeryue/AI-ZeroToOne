// Explicit template instantiations for MCTSTree<N>.

#include "mcts.h"

namespace mcts {

template class MCTSTree<9>;
template class MCTSTree<13>;
template class MCTSTree<19>;

}  // namespace mcts
