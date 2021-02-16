version 1.0

task run_mighty_codes_sa_bac {

    input {
        File base_yaml_file
        File mighty_codes_tar_gz

        # optional overrides
        Int? n_subsystems
        Int? eval_split_size
        Int? max_iters
        Int? max_cycles
        Int? perturb_max_hamming_distance
        Int? perturb_max_hamming_weight_change
        Int? perturb_max_neighbor_hop_range
        Int? checkpoint_interval_seconds
        
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
         docker: "us.gcr.io/broad-dsde-methods/pyro_matplotlib:0.0.7"
         bootDiskSizeGb: 100
         memory: "26G"
         cpu: 4
         zones: "us-east1-d us-east1-c"
         gpuCount: 1
         gpuType: "nvidia-tesla-p100"
         maxRetries: 10
         preemptible_tries: 10
         preemptible: 10
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
