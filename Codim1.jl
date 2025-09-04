"""
    Codim1

A Julia interface for the Codim1 code written by Robert Melville. The Julia
interface follows the interface of the C code, though user functions are not
currently supported.

It relies on the C code compiled as a shared library. Compile the shared library with

- Mac OS: `gcc codim1.c -lblas -llapack -lm -fPIC -shared -o libcodim1.dylib`
- Linux: `gcc codim1.c -lblas -llapack -lm -fPIC -shared -o libcodim1.so`

It should be possible to run under Windows as well, but that is not tested.

Example code is provided in the `helix_example` function.

Written by David A.W. Barton (david.barton@bristol.ac.uk) 2025 and released
under the [Common Public License](https://opensource.org/license/cpl1-0-txt).
"""
module Codim1

export codim1!, allocate_mem, check_start_point, improve_start_point, find_transverse, path_start_a, path_start_b, path_advance, free_mem

# Path to the shared library (adjust as needed)
const LIBCODIM1 = joinpath(@__DIR__, ifelse(Sys.isapple(), "libcodim1.dylib", "libcodim1.so"))

@enum SimpconTask begin
    allocate_mem = 0
    check_start_point = 1
    improve_start_point = 2
    find_transverse = 3
    path_start_a = 4
    path_start_b = 5
    path_advance = 6
    free_mem = 7
end

# This is type unstable and will result in dynamic dispatch (slower)
const RESIDUAL_FCN! = Ref{Any}(nothing)

# Alternatively, this can be made type stable and faster using FunctionWrappers.jl

# using FunctionWrappers: FunctionWrapper
# const RESIDUAL_FCN! = Ref{FunctionWrapper{Nothing, Tuple{Vector{Float64}, Vector{Float64}}}}((res, x) -> nothing)

function residual_wrapper(n::Cint, nu::Cint, _x::Ptr{Cdouble}, _rsd::Ptr{Cdouble}, u::Ptr{Cdouble}, param::Ptr{Cvoid})::Cvoid
    x = unsafe_wrap(Array{Float64}, _x, n + 1)
    rsd = unsafe_wrap(Array{Float64}, _rsd, n)
    RESIDUAL_FCN![](rsd, x)
    return nothing
end

const C_RESIDUAL = @cfunction(residual_wrapper, Cvoid, (Cint, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cvoid}))

"""
    codim1!(y::Vector{Float64}, grain::Float64, cum_al::Ref{Float64}, io, task, residual!, param = nothing)

This function follows the interface of the corresponding C code with the following exceptions.

- No vector lengths are required since they can be inferred from the vectors themselves.
- User functions are not (currently) supported.
- `io` should be a regular file object or `stdout`.
"""
function codim1!(y::Vector{Float64}, grain::Float64, cum_al::Ref{Float64}, io, task::SimpconTask, residual!, param = nothing)
    # Infer lengths from the arrays passed
    n = Cint(length(y) - 1)
    nu = Cint(0)

    # Get the C file pointer from an existing IO
    _io = io === stdout ? so = Libc.FILE(Libc.RawFD(1), "w") : Libc.FILE(io)

    # Convert param to void pointer
    param_ptr = param === nothing ? C_NULL : pointer_from_objref(param)

    # Store the residual function for use in residual_wrapper
    RESIDUAL_FCN![] = residual!

    @ccall LIBCODIM1.codim1(n::Cint, nu::Cint, y::Ptr{Cdouble}, Cdouble[]::Ptr{Cdouble}, grain::Cdouble, cum_al::Ref{Cdouble}, _io::Ptr{Cvoid}, task::Cint, C_RESIDUAL::Ptr{Nothing}, param_ptr::Ptr{Cvoid})::Cvoid

    return nothing
end

function helix_example()
    function residual!(res, x)
        # Helix example
        a = 0.65
        b = 1.35

        res[1] = x[1] * sin(x[3]) - x[2] * cos(x[3])
        res[2] = (x[1] * x[1]) / (a * a) + (x[2] * x[2]) / (b * b) - 1.0

        return nothing
    end

    io = stdout

    # Initial point
    y = [0.42, 0.92, 1.122073]

    # Parameters
    grain = 0.25
    cum_al = Ref(0.0)

    # Initialize and improve starting point
    println("Allocating memory...")
    codim1!(y, grain, cum_al, io, allocate_mem, residual!)

    println("Checking start point...")
    codim1!(y, grain, cum_al, io, check_start_point, residual!)

    println("Improving start point...")
    codim1!(y, grain, cum_al, io, improve_start_point, residual!)
    codim1!(y, grain, cum_al, io, improve_start_point, residual!)

    println("Final check of start point...")
    codim1!(y, grain, cum_al, io, check_start_point, residual!)

    # Find transverse direction and start path
    println("Finding transverse direction...")
    codim1!(y, grain, cum_al, io, find_transverse, residual!)

    println("Starting path...")
    codim1!(y, grain, cum_al, io, path_start_a, residual!)

    # Store trajectory points
    trajectory = Vector{Vector{Float64}}()
    push!(trajectory, copy(y))

    println("Initial point: $(y[1]) $(y[2]) $(y[3])")

    # Continue along the path
    println("Continuing along path...")
    step_count = 0
    max_steps = 1000  # Safety limit

    while cum_al[] < 20.0 && step_count < max_steps
        codim1!(y, grain, cum_al, io, path_advance, residual!)
        push!(trajectory, copy(y))

        if step_count % 10 == 0  # Print every 10th step
            println("Step $step_count, arc length: $(cum_al[]), point: $(y[1]) $(y[2]) $(y[3])")
        end

        step_count += 1
    end

    println("Path continuation completed. Total steps: $step_count")
    println("Final arc length: $(cum_al[])")
    println("Final point: $(y[1]) $(y[2]) $(y[3])")

    # Free memory
    println("Freeing memory...")
    codim1!(y, grain, cum_al, io, free_mem, residual!)

    return trajectory
end

end # module
