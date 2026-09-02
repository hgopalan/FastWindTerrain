#ifndef FWT_ERROR_H_
#define FWT_ERROR_H_

#include <functional>
#include <stdexcept>
#include <string>

namespace fwt {

// -----------------------------------------------------------------------
// How the solver reports a bad input, in a way that works for both
// entry points.
//
// THE PROBLEM. amrex::Abort kills the process. That is exactly right for
// a command-line run -- a grid that does not reach the domain top must
// not quietly produce a plotfile -- and exactly wrong inside Python,
// where it takes the interpreter down and loses whatever else the
// session was holding. A dataset generator that dies on sample 700
// because sample 700's parameters were out of range should raise, so the
// loop can record it and carry on.
//
// THE SPLIT. A bad input throws InputError. The executable catches it in
// main() and aborts with the same message, so its behaviour -- the
// printed diagnostic, the nonzero exit, the absence of a report file --
// is unchanged and the regtests that assert those still pass. The
// bindings let it become a Python exception.
//
// This covers USER INPUT only. An internal invariant -- a box split in z
// when the decomposition promised it would not -- stays an assertion:
// it is a bug in this code, not something a caller can provoke, and it
// should abort loudly wherever it happens.
// -----------------------------------------------------------------------
class InputError : public std::runtime_error
{
public:
    explicit InputError (const std::string& what) : std::runtime_error(what) {}
};

// -----------------------------------------------------------------------
// Non-fatal diagnostics take the same shape. A stretched grid that
// overshoots the requested domain height is a warning: the run
// continues, having adjusted the top.
//
// The default handler prints exactly what the code printed before this
// existed, so the executable's output is byte-for-byte what it was. The
// bindings install one that calls Python's warnings.warn, so a notebook
// sees a real warning it can filter, promote to an error, or capture --
// rather than a line of text on stdout it will never look at.
//
// Handlers are process-global, like the thing they are standing in for.
// -----------------------------------------------------------------------
using WarningHandler = std::function<void(const std::string&)>;

// Installs a handler and returns the previous one, so a caller can
// restore it. Passing an empty function restores the default.
WarningHandler SetWarningHandler (WarningHandler handler);

// Emits a warning through the installed handler. The message is the
// complete text, newlines included, exactly as it should appear.
void Warn (const std::string& message);

} // namespace fwt

#endif
