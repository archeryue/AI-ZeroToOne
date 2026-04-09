// Explicit template instantiations for SelfPlayWorker.
#include "worker.h"

namespace alphazero {
template class SelfPlayWorker<9>;
template class SelfPlayWorker<13>;
template class SelfPlayWorker<19>;
}
