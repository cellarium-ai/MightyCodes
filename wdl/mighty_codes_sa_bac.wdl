version 1.0

task run_mighty_codes_sa_bac {

    input {
        # runtime
        String docker_image
        Int hardware_boot_disk_size
        Int hardware_memory
        Int hardware_cpu_count
        String hardware_zones
        String hardware_gpu_type
        Int hardware_preemptible_tries
        
        # base
        File base_yaml_file
        File mighty_codes_tar_gz

        # required overrides
        Int quality_factor
        String experiment_prefix
        String channel_model
        Int min_hamming_weight
        Int max_hamming_weight
        Int code_length
        Int n_types
        Float source_nonuniformity
        String metric_type

        # optional overrides
        Int? n_subsystems
        Int? eval_split_size
        Int? max_iters
        Int? max_cycles
        Float? convergence_abs_tol
        Int? convergence_countdown
        Int? perturb_max_hamming_distance
        Int? perturb_max_hamming_weight_change
        Int? perturb_max_neighbor_hop_range
        Int? checkpoint_interval_seconds
    }

    command <<<
        
        # extract MightyCodes
        tar xvzf ~{mighty_codes_tar_gz}
        
        # install MightyCodes
        pip install -e MightyCodes/
        
        # update the input YAML file
        mighty-codes yaml-tools \
            --input-yaml-file ~{base_yaml_file} \
            --output-yaml-file updated_params.yaml \
            ~{"--update n_subsystems " + n_subsystems + " int"} \
            ~{"--update eval_split_size " + eval_split_size + " int"} \
            ~{"--update max_iters " + max_iters + " int"} \
            ~{"--update max_cycles " + max_cycles + " int"} \
            ~{"--update convergence_abs_tol " + convergence_abs_tol + " float"} \
            ~{"--update convergence_countdown " + convergence_countdown + " int"} \
            ~{"--update perturb_max_hamming_distance " + perturb_max_hamming_distance + " int"} \
            ~{"--update perturb_max_hamming_weight_change " + perturb_max_hamming_weight_change + " int"} \
            ~{"--update perturb_max_neighbor_hop_range " + perturb_max_neighbor_hop_range + " int"} \
            ~{"--update checkpoint_interval_seconds " + checkpoint_interval_seconds + " int"} \
            ~{"--update quality_factor " + quality_factor + " int"} \
            ~{"--update experiment_prefix " + experiment_prefix + " str"} \
            ~{"--update channel_model " + channel_model + " str"} \
            ~{"--update min_hamming_weight " + min_hamming_weight + " int"} \
            ~{"--update max_hamming_weight " + max_hamming_weight + " int"} \
            ~{"--update code_length " + code_length + " int"} \
            ~{"--update n_types " + n_types + " int"} \
            ~{"--update source_nonuniformity " + source_nonuniformity + " float"} \
            ~{"--update metric_type " + metric_type + " str"}
            
       # run sa-bac
       mighty-codes sa-bac --input-yaml-file updated_params.yaml
        
    >>>
    
    runtime {
         docker: "${docker_image}"
         bootDiskSizeGb: hardware_boot_disk_size
         memory: "${hardware_memory}G"
         cpu: hardware_cpu_count
         zones: "${hardware_zones}"
         gpuCount: 1
         gpuType: "${hardware_gpu_type}"
         maxRetries: 0
         preemptible: hardware_preemptible_tries
         checkpointFile: "checkpoint.tar.gz"
    }

    output {
        File final_state_tar_gz = "final_state.tar.gz"
    }
    
}

workflow mighty_codes_sa_bac_workflow {
  
  call run_mighty_codes_sa_bac

  output {
    File final_state_tar_gz = run_mighty_codes_sa_bac.final_state_tar_gz
  }
  
}
