/// \file dmrg_results_json.c
/// \brief Write DMRG results to JSON (see dmrg_results_json.h and DMRG_RESULTS_JSON_SCHEMA.md).

#include "dmrg_results_json.h"
#include <stdio.h>

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
