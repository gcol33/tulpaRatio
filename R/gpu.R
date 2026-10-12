#' GPU Support Utilities
#'
#' Functions to check GPU availability and capabilities for accelerated
#' GP computations.
#'
#' @examples
#' gpu_available()
#' ratiod_threads()
#' @name gpu-support
NULL

#' Check Whether GPU Support Is Available
#'
#' @description
#' Checks whether tulpaRatio was compiled with GPU support (CUDA or OpenCL)
#' and whether compatible GPU hardware is detected.
#'
#' @return Logical; TRUE if GPU acceleration is available, FALSE otherwise.
#'
#' @details
#' GPU support requires:
#' 1. tulpaRatio compiled with GPU support (`-DUSE_CUDA` or `-DUSE_OPENCL`)
#' 2. Compatible GPU hardware (NVIDIA for CUDA, any GPU for OpenCL)
#' 3. Appropriate drivers installed
#'
#' The `solver = "gpu"` option in [spatial_gp()] will automatically fall back
#' to PCG if GPU support is unavailable.
#'
#' @examples
#' # Check if GPU is available
#' gpu_available()
#'
#' # Get detailed GPU info
#' if (gpu_available()) {
#'   gpu_info()
#' }
#'
#' @seealso [spatial_gp()] for using GPU-accelerated GP models
#' @export
gpu_available <- function() {
  # Check if compiled with GPU support
  tryCatch({
    cpp_gpu_available()
  }, error = function(e) {
    FALSE
  })
}

#' GPU Device Information
#'
#' @description
#' Returns information about available GPU devices including name, memory,
#' compute capability, and backend (CUDA or OpenCL).
#'
#' @return A list with GPU information, or NULL if no GPU is available.
#'   \describe{
#'     \item{available}{Logical; whether GPU is available}
#'     \item{backend}{Character; "cuda", "opencl", or "none"}
#'     \item{device_count}{Integer; number of GPU devices}
#'     \item{devices}{List of device info (name, memory, compute capability)}
#'   }
#'
#' @examples
#' info <- gpu_info()
#' if (!is.null(info) && info$available) {
#'   cat("GPU backend:", info$backend, "\n")
#'   cat("Devices:", info$device_count, "\n")
#' }
#'
#' @export
gpu_info <- function() {
  if (!gpu_available()) {
    return(list(
      available = FALSE,
      backend = "none",
      device_count = 0L,
      devices = list()
    ))
  }

  tryCatch({
    cpp_gpu_info()
  }, error = function(e) {
    list(
      available = FALSE,
      backend = "none",
      device_count = 0L,
      devices = list(),
      error = conditionMessage(e)
    )
  })
}

#' @rdname gpu_info
#' @param x GPU info object from [gpu_info()]
#' @param ... Ignored
#' @export
print.tulpaRatio_gpu_info <- function(x, ...) {
  cat("tulpaRatio GPU Support\n")
  cat("====================\n\n")

  if (!x$available) {
    cat("Status: Not available\n")
    if (!is.null(x$error)) {
      cat("Error:", x$error, "\n")
    }
    cat("\nTo enable GPU support:\n")
    cat("  CUDA: Install CUDA toolkit, rebuild with -DUSE_CUDA\n")
    cat("  OpenCL: Install OpenCL SDK, rebuild with -DUSE_OPENCL\n")
    return(invisible(x))
  }

  cat("Status: Available\n")
  cat("Backend:", toupper(x$backend), "\n")
  cat("Devices:", x$device_count, "\n\n")

  for (i in seq_along(x$devices)) {
    dev <- x$devices[[i]]
    cat(sprintf("Device %d: %s\n", i - 1, dev$name))
    cat(sprintf("  Memory: %.1f GB\n", dev$memory_mb / 1024))
    if (!is.null(dev$compute_capability)) {
      cat(sprintf("  Compute: %s\n", dev$compute_capability))
    }
    cat("\n")
  }

  invisible(x)
}
