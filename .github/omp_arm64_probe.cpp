#include <omp.h>
#include <cmath>
#include <cstdio>
#include <vector>

static double term(int y, double mu, bool gam) {
  double v = y * std::log(mu) - mu;
  if (gam) v -= std::lgamma(y + 1.0);
  return v;
}

int main() {
  const int N = 200;
  std::vector<int> y(N);
  std::vector<double> mu(N);
  unsigned s = 12345u;
  for (int i = 0; i < N; i++) {
    s = s * 1664525u + 1013904223u;
    y[i] = (s >> 16) % 12;
    s = s * 1664525u + 1013904223u;
    mu[i] = 2.0 + 6.0 * ((s >> 8) & 0xffff) / 65535.0;
  }
  std::printf("procs=%d max_threads=%d\n", omp_get_num_procs(), omp_get_max_threads());

  for (int gam = 0; gam < 2; gam++) {
    double serial = 0.0;
    for (int i = 0; i < N; i++) serial += term(y[i], mu[i], gam);
    for (int nt : {1, 2, 4}) {
      int bad = 0;
      double worst = 0.0;
      for (int rep = 0; rep < 500; rep++) {
        double ll = 0.0;
#pragma omp parallel for reduction(+ : ll) schedule(static) num_threads(nt)
        for (int i = 0; i < N; i++) ll += term(y[i], mu[i], gam);
        if (std::fabs(ll - serial) > 1e-8) {
          bad++;
          worst = std::fmax(worst, std::fabs(ll / serial - 1.0));
        }
      }
      std::printf("lgamma=%d reduction threads=%d mismatches=%d/500 worst_rel=%.4f\n",
                  gam, nt, bad, worst);
    }
  }

  for (int nt : {2, 4}) {
    int bad = 0, short_team = 0;
    std::vector<double> out(N);
    for (int rep = 0; rep < 500; rep++) {
      int team = 0;
#pragma omp parallel num_threads(nt)
      {
#pragma omp single
        team = omp_get_num_threads();
#pragma omp for schedule(static)
        for (int i = 0; i < N; i++) out[i] = std::lgamma(y[i] + 1.0);
      }
      if (team != nt) short_team++;
      for (int i = 0; i < N; i++)
        if (std::fabs(out[i] - std::lgamma(y[i] + 1.0)) > 1e-12) { bad++; break; }
    }
    std::printf("lgamma slots threads=%d bad_reps=%d short_team_reps=%d\n", nt, bad, short_team);
  }

  for (int nt : {2, 4}) {
    int bad = 0;
    for (int rep = 0; rep < 500; rep++) {
      std::vector<double> part(nt, 0.0);
      double serial = 0.0;
      for (int i = 0; i < N; i++) serial += term(y[i], mu[i], 1);
#pragma omp parallel num_threads(nt)
      {
        const int t = omp_get_thread_num();
        const int T = omp_get_num_threads();
        double acc = 0.0;
        for (int i = t * N / T; i < (t + 1) * N / T; i++) acc += term(y[i], mu[i], 1);
        part[t] = acc;
      }
      double tot = 0.0;
      for (double v : part) tot += v;
      if (std::fabs(tot - serial) > 1e-8) bad++;
    }
    std::printf("manual partials threads=%d bad_reps=%d/500\n", nt, bad);
  }
  return 0;
}
