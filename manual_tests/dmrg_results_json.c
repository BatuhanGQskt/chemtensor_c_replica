/// \file dmrg_results_json.c
/// \brief Write DMRG results to JSON (see dmrg_results_json.h and DMRG_RESULTS_JSON_SCHEMA.md).

#include "dmrg_results_json.h"
#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>

static int ensure_parent_dirs(const char* path)
{
	// Create parent directories for `path` if needed (mkdir -p behavior).
	// This is intentionally lightweight and only handles '/' separators (Linux/WSL).
	if (!path || path[0] == '\0')
		return 0;

	char buf[4096];
	size_t n = strlen(path);
	if (n >= sizeof(buf))
		return -1;
	memcpy(buf, path, n + 1);

	char* last_slash = strrchr(buf, '/');
	if (!last_slash)
		return 0; // current directory
	if (last_slash == buf)
		return 0; // path like "/file.json" -> root exists

	*last_slash = '\0';

	// Iteratively create each component.
	for (char* p = buf + 1; *p; p++) {
		if (*p != '/')
			continue;
		*p = '\0';
		if (mkdir(buf, 0777) != 0 && errno != EEXIST)
			return -1;
		*p = '/';
	}
	if (mkdir(buf, 0777) != 0 && errno != EEXIST)
		return -1;

	return 0;
}

int dmrg_results_to_json(
	const char* path,
	const char* impl,
	const char* model,
	const struct dmrg_json_params* params,
	double energy_final,
	const double* en_sweeps,
	int num_sweeps,
	const double* entropy,
	const long* bond_dims,
	int num_bond_dims,
	double norm)
{
	if (ensure_parent_dirs(path) != 0)
		return -1;

	FILE* f = fopen(path, "w");
	if (!f)
		return -1;

	fprintf(f, "{\n");
	fprintf(f, "  \"impl\": \"%s\",\n", impl);
	fprintf(f, "  \"model\": \"%s\",\n", model);
	fprintf(f, "  \"params\": {\n");
	fprintf(f, "    \"nsites\": %d,\n", params->nsites);
	fprintf(f, "    \"d\": %d,\n", params->d);
	fprintf(f, "    \"J\": %.15g,\n", params->J);
	fprintf(f, "    \"D\": %.15g,\n", params->D);
	fprintf(f, "    \"h\": %.15g,\n", params->h);
	fprintf(f, "    \"num_sweeps\": %d,\n", params->num_sweeps);
	fprintf(f, "    \"maxiter_lanczos\": %d,\n", params->maxiter_lanczos);
	fprintf(f, "    \"chi_max\": %d", params->chi_max);
	if (params->tol_split >= 0.0) {
		fprintf(f, ",\n    \"tol_split\": %.15g", params->tol_split);
	}
	fprintf(f, "\n  },\n");
	fprintf(f, "  \"energy_final\": %.15g,\n", energy_final);

	fprintf(f, "  \"en_sweeps\": [");
	for (int i = 0; i < num_sweeps; i++) {
		fprintf(f, "%.15g%s", en_sweeps[i], i < num_sweeps - 1 ? ", " : "");
	}
	fprintf(f, "],\n");

	if (entropy && params->nsites > 1) {
		int num_entropy = params->nsites - 1;
		fprintf(f, "  \"entropy\": [");
		for (int i = 0; i < num_entropy; i++) {
			fprintf(f, "%.15g%s", entropy[i], i < num_entropy - 1 ? ", " : "");
		}
		fprintf(f, "],\n");
	}

	fprintf(f, "  \"bond_dims\": [");
	for (int i = 0; i < num_bond_dims; i++) {
		fprintf(f, "%ld%s", bond_dims[i], i < num_bond_dims - 1 ? ", " : "");
	}
	fprintf(f, "],\n");

	fprintf(f, "  \"norm\": %.15g\n", norm);
	fprintf(f, "}\n");

	if (fclose(f) != 0)
		return -1;
	return 0;
}
