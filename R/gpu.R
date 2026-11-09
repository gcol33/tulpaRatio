#' GPU Support Utilities
#'
#' Functions to check whether CUDA or OpenCL driver libraries can be loaded.
#'
#' @examples
#' gpu_available()
#' ratiod_threads()
#' @name gpu-support
NULL

#' Check Whether GPU Support Is Available
#'
#' @description
#' Checks whether a CUDA or OpenCL driver library can be loaded at runtime.
#'
#' @return Logical; TRUE if a driver library can be loaded, FALSE otherwise.
#'
#' @details
#' This reports driver-library availability. Model fitting uses CPU solvers.
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
#' @seealso [gpu_info()]
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
#' Reports the loadable driver backend (CUDA or OpenCL). Device details are
#' placeholders: a loadable driver is reported as one device with unknown
#' memory and compute capability.
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
  cat_title("tulpaRatio GPU Support")

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
