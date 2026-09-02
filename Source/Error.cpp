#include "Error.H"

#include <AMReX_Print.H>

namespace fwt {

namespace {

// The default: print exactly what the code printed before the handler
// existed. amrex::Print writes on the IO processor only, which is what
// keeps a multi-rank run from repeating the same warning N times.
void DefaultWarningHandler (const std::string& message)
{
    amrex::Print() << message;
}

WarningHandler& Handler ()
{
    static WarningHandler s_handler = DefaultWarningHandler;
    return s_handler;
}

} // namespace

WarningHandler SetWarningHandler (WarningHandler handler)
{
    WarningHandler previous = Handler();
    Handler() = handler ? std::move(handler) : WarningHandler(DefaultWarningHandler);
    return previous;
}

void Warn (const std::string& message)
{
    Handler()(message);
}

} // namespace fwt
