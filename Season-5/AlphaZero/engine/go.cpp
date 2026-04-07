// Explicit template instantiations for Board<N> and Game<N>.

#include "go.h"

namespace go {

template struct Board<9>;
template struct Board<13>;
template struct Board<19>;

template struct Game<9>;
template struct Game<13>;
template struct Game<19>;

}  // namespace go
