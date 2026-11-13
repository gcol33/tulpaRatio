test_that("no gradient function keeps its scratch in the shared ModelData", {
    skip_on_cran()
    skip_if(tulpaRatio:::cpp_num_procs() < 4, "needs 4 cores to interleave chains")
    fields <- tulpaRatio:::cpp_gradient_fields()
    expect_gt(length(fields), 50)
    dev <- vapply(fields, function(f) {
        tulpaRatio:::cpp_gradient_race(f, family = gradient_fixture_family(f),
                                    n_points = 8L, n_threads = 4L, rounds = 8L)
    }, numeric(1))
    racing <- names(dev)[dev != 0]
    expect_identical(racing, character(0), info = paste0("these gradients differ between a serial and a concurrent pass over one ",
        "ModelData, so they write scratch that every chain thread shares:\n", paste(sprintf("  %s: max deviation %.3e",
            racing, dev[racing]), collapse = "\n")))
})

test_that("the race probe measures the SVC gradient it was written for (#30)", {
    skip_on_cran()
    skip_if(tulpaRatio:::cpp_num_procs() < 4, "needs 4 cores to interleave chains")
    expect_equal(tulpaRatio:::cpp_gradient_race("svc", n_points = 8L, n_threads = 4L, rounds = 16L),
        0)
    g <- tulpaRatio:::cpp_gradient_check("svc")
    expect_true(all(is.finite(g$analytic)))
    expect_gt(max(abs(g$analytic)), 0)
})
