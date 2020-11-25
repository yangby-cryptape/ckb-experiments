use std::collections::HashMap;

use ckb_chain_spec::{
    calculate_block_reward,
    consensus::{build_genesis_epoch_ext, Consensus, ConsensusBuilder},
};
use ckb_dao_utils::genesis_dao_data_with_satoshi_gift;
use ckb_types::{
    bytes,
    core::{self, Capacity},
    h160, packed,
    prelude::*,
    utilities::{compact_to_difficulty, difficulty_to_compact},
    H160, U256,
};

const INITIAL_PRIMARY_EPOCH_REWARD: Capacity = Capacity::shannons(191_780_821_917_808);
const DEFAULT_SECONDARY_EPOCH_REWARD: Capacity = Capacity::shannons(61_369_863_013_698);
const DEFAULT_EPOCH_DURATION_TARGET: u64 = 4 * 60 * 60; // 4 hours, unit: second

const SATOSHI_PUBKEY_HASH: H160 = h160!("0x62e907b15cbf27d5425399ebf6f0fb50ebb88f18");
const SATOSHI_CELL_OCCUPIED_RATIO: core::Ratio = core::Ratio(6, 10);

struct Chain {
    blocks: Vec<core::BlockView>,
    epochs: Vec<core::EpochExt>,
    uncles: Vec<usize>,
    index: HashMap<packed::Byte32, core::BlockNumber>,
}

fn build_chain(genesis_epoch_length: core::BlockNumber, compact_target: u32) -> (Chain, Consensus) {
    let cellbase = {
        let input = packed::CellInput::new_cellbase_input(0);
        let output = {
            let empty_output = packed::CellOutput::new_builder().build();
            let occupied = empty_output.occupied_capacity(Capacity::zero()).unwrap();
            empty_output.as_builder().capacity(occupied.pack()).build()
        };
        let witness = packed::Script::default().into_witness();
        core::TransactionBuilder::default()
            .input(input)
            .output(output)
            .output_data(bytes::Bytes::new().pack())
            .witness(witness)
            .build()
    };
    let epoch_ext = build_genesis_epoch_ext(
        INITIAL_PRIMARY_EPOCH_REWARD,
        compact_target,
        genesis_epoch_length,
        DEFAULT_EPOCH_DURATION_TARGET,
    );
    let genesis_block = {
        let primary_issuance =
            calculate_block_reward(INITIAL_PRIMARY_EPOCH_REWARD, genesis_epoch_length);
        let secondary_issuance =
            calculate_block_reward(DEFAULT_SECONDARY_EPOCH_REWARD, genesis_epoch_length);
        let dao = genesis_dao_data_with_satoshi_gift(
            vec![&cellbase],
            &SATOSHI_PUBKEY_HASH,
            SATOSHI_CELL_OCCUPIED_RATIO,
            primary_issuance,
            secondary_issuance,
        )
        .unwrap();
        core::BlockBuilder::default()
            .compact_target(compact_target.pack())
            .dao(dao)
            .transaction(cellbase)
            .build()
    };
    let chain = {
        #[allow(clippy::mutable_key_type)]
        let mut index = HashMap::new();
        index.insert(genesis_block.hash(), genesis_block.number());
        Chain {
            blocks: vec![genesis_block.clone()],
            epochs: vec![epoch_ext.clone()],
            uncles: vec![0],
            index,
        }
    };
    let consensus = ConsensusBuilder::new(genesis_block, epoch_ext)
        .initial_primary_epoch_reward(INITIAL_PRIMARY_EPOCH_REWARD)
        .build();
    (chain, consensus)
}

fn cast_u256_to_u64(name: &str, value: &U256) -> u64 {
    if value.0[1] == 0 && value.0[2] == 0 && value.0[3] == 0 {
        value.0[0]
    } else {
        panic!("{} is too large", name);
    }
}

fn run_chain(
    stabilized_length: core::BlockNumber,
    stabilized_compact_target: u32,
    stabilized_hash_rate: U256,
) {
    let mut prev_compact_target = stabilized_compact_target;
    let (mut chain, consensus) = build_chain(stabilized_length, prev_compact_target);
    // Get the estimated genesis hash rate.
    let mut prev_epoch_hash_rate = {
        chain.epochs[chain.epochs.len() - 1]
            .previous_epoch_hash_rate()
            .clone()
    };

    loop {
        {
            // Calculate which blocks are left in last epoch.
            let (start, end, duration) = {
                let last_epoch = chain.epochs[chain.epochs.len() - 1].clone();
                let last_block = chain.blocks[chain.blocks.len() - 1].clone();
                let start = last_block.number();
                let end = last_epoch.start_number() + last_epoch.length() - 1;
                // Use the stabilized hash rate to calculate the duration between two continuous blocks.
                let duration = {
                    let length = {
                        // https://github.com/nervosnetwork/ckb/blob/v0.38.0/spec/src/consensus.rs#L196-L199
                        let length_u256 =
                            (&stabilized_hash_rate) * DEFAULT_EPOCH_DURATION_TARGET * 40u32
                                / compact_to_difficulty(last_epoch.compact_target())
                                / 41u32;
                        cast_u256_to_u64("length", &length_u256)
                    };
                    if last_epoch.is_genesis() {
                        (DEFAULT_EPOCH_DURATION_TARGET * 1000) / (length - 1)
                    } else {
                        (DEFAULT_EPOCH_DURATION_TARGET * 1000) / length
                    }
                };
                (start, end, duration)
            };
            // Fill up the last epoch.
            for idx in start..end {
                let idx = idx as usize;
                assert!(chain.blocks.len() == idx + 1);
                assert!(chain.blocks.len() == chain.index.len());
                assert!(chain.blocks.len() == chain.uncles.len());
                let last_block = chain.blocks[idx].clone();
                let uncles_cnt = chain.uncles[idx];
                let timestamp = last_block.timestamp() + duration;
                let mut next_block_builder = last_block
                    .as_advanced_builder()
                    .timestamp(timestamp.pack())
                    .number((last_block.number() + 1).pack())
                    .parent_hash(last_block.hash());
                // We assume that the orphan rate was in ideal state (2.5% == 1/40).
                if idx % 40 == 1 {
                    next_block_builder =
                        next_block_builder.uncle(core::BlockBuilder::default().build().as_uncle());
                    chain.uncles.push(uncles_cnt + 1);
                } else {
                    chain.uncles.push(uncles_cnt);
                }
                let next_block = next_block_builder.build();
                chain.index.insert(next_block.hash(), next_block.number());
                chain.blocks.push(next_block);
            }
        }
        // Go to next epoch.
        {
            let last_epoch = chain.epochs[chain.epochs.len() - 1].clone();
            let last_block = chain.blocks[chain.blocks.len() - 1].clone();

            let next_epoch = {
                let get_block_header = |hash: &packed::Byte32| {
                    let num_opt = chain.index.get(hash).copied();
                    if let Some(num) = num_opt {
                        Some(chain.blocks[num as usize].header())
                    } else {
                        panic!("could not find header");
                    }
                };
                let total_uncles_count = |hash: &packed::Byte32| {
                    let num_opt = chain.index.get(hash).copied();
                    if let Some(num) = num_opt {
                        let cnt = chain.uncles[num as usize] as u64;
                        Some(cnt)
                    } else {
                        panic!("could not count uncles");
                    }
                };
                consensus.next_epoch_ext(
                    &last_epoch,
                    &last_block.header(),
                    get_block_header,
                    total_uncles_count,
                )
            };

            if let Some(next_epoch) = next_epoch {
                let hash_rate_change = {
                    let (sign, value_u256) =
                        if next_epoch.previous_epoch_hash_rate() > &prev_epoch_hash_rate {
                            (
                                1,
                                next_epoch.previous_epoch_hash_rate() - &prev_epoch_hash_rate,
                            )
                        } else {
                            (
                                -1,
                                &prev_epoch_hash_rate - next_epoch.previous_epoch_hash_rate(),
                            )
                        };
                    sign * (cast_u256_to_u64("hash-rate-change", &value_u256) as i64)
                };
                let prev_epoch_hash_rate_f64 =
                    cast_u256_to_u64("prev-epoch-hash-rate", &prev_epoch_hash_rate) as f64;
                let next_prev_epoch_hash_rate_f64 = cast_u256_to_u64(
                    "next-prev-epoch-hash-rate",
                    next_epoch.previous_epoch_hash_rate(),
                ) as f64;
                let compact_target_change =
                    i64::from(next_epoch.compact_target()) - i64::from(prev_compact_target);
                let compact_target_change_precent = (f64::from(next_epoch.compact_target())
                    - f64::from(prev_compact_target))
                    * 100.0
                    / f64::from(prev_compact_target);
                println!(
                    "epoch[{:4}]: {{ length: {:#4} ({:+#5}), compact-target: {:#16} ({:+6.2}%, {:+#8}), hash-rate {:#16} ({:+6.2}%, {:+#8}) }}",
                    next_epoch.number(),
                    next_epoch.length(),
                    next_epoch.length() as i64 - stabilized_length as i64,
                    next_epoch.compact_target(),
                    compact_target_change_precent,
                    compact_target_change,
                    next_prev_epoch_hash_rate_f64,
                    (hash_rate_change as f64) * 100.0 / prev_epoch_hash_rate_f64,
                    hash_rate_change,
                );
                prev_compact_target = next_epoch.compact_target();
                prev_epoch_hash_rate = next_epoch.previous_epoch_hash_rate().clone();
                chain.epochs.push(next_epoch);
            } else {
                panic!("next epoch is none");
            }
        }
    }
}

fn main() {
    let matches = clap::App::new("CKB Epoch Convergence")
        .version(clap::crate_version!())
        .author(clap::crate_authors!("\n"))
        .about("Test epoch convergence when the hash rate is stabilized.")
        .arg(
            clap::Arg::with_name("epoch-length")
                .long("epoch-length")
                .takes_value(true)
                .required(false),
        )
        .arg(
            clap::Arg::with_name("compact-target")
                .long("compact-target")
                .takes_value(true)
                .required(false),
        )
        .arg(
            clap::Arg::with_name("hash-rate")
                .long("hash-rate")
                .takes_value(true)
                .required(false),
        )
        .get_matches();
    let epoch_length_opt = matches.value_of("epoch-length");
    let compact_target_opt = matches.value_of("compact-target");
    let hash_rate_opt = matches.value_of("hash-rate");
    let (epoch_length, compact_target, hash_rate) = match (
        epoch_length_opt,
        compact_target_opt,
        hash_rate_opt,
    ) {
        (Some(epoch_length), Some(compact_target), None) => {
            let epoch_length: u64 = epoch_length.parse().unwrap();
            let compact_target: u32 = compact_target.parse().unwrap();
            println!("input epoch-length  : {:#16}", epoch_length);
            println!("input compact-target: {:#16}", compact_target);
            let hash_rate = {
                let hash_rate_u256 = compact_to_difficulty(compact_target) * 41u32 * epoch_length
                    / 40u32
                    / DEFAULT_EPOCH_DURATION_TARGET;
                cast_u256_to_u64("arg-hash-rate", &hash_rate_u256)
            };
            if hash_rate == 0 {
                panic!("compact-target is too small");
            }
            println!("   so hash-rate     : {:#16}", hash_rate);
            (epoch_length, compact_target, hash_rate)
        }
        (Some(epoch_length), None, Some(hash_rate)) => {
            let epoch_length: u64 = epoch_length.parse().unwrap();
            let hash_rate: u64 = hash_rate.parse().unwrap();
            println!("input epoch-length  : {:#16}", epoch_length);
            println!("input hash-rate     : {:#16}", hash_rate);
            let compact_target = {
                let difficulty = U256::from(hash_rate) * DEFAULT_EPOCH_DURATION_TARGET * 40u32
                    / 41u32
                    / epoch_length;
                difficulty_to_compact(difficulty)
            };
            println!("   so compact-target: {:#16}", compact_target);
            (epoch_length, compact_target, hash_rate)
        }
        (None, Some(compact_target), Some(hash_rate)) => {
            let compact_target: u32 = compact_target.parse().unwrap();
            let hash_rate: u64 = hash_rate.parse().unwrap();
            let epoch_length = {
                let length_u256 = (&U256::from(hash_rate)) * DEFAULT_EPOCH_DURATION_TARGET * 40u32
                    / compact_to_difficulty(compact_target)
                    / 41u32;
                cast_u256_to_u64("arg-epoch-length", &length_u256)
            };
            println!("input compact-target: {:#16}", compact_target);
            println!("input hash-rate     : {:#16}", hash_rate);
            println!("   so epoch-length  : {:#16}", epoch_length);
            (epoch_length, compact_target, hash_rate)
        }
        _ => {
            panic!("epoch-length, compact-target, hash-rate: two and only two of them should be provided");
        }
    };
    if epoch_length > 1800 || epoch_length < 300 {
        panic!("unsupported genesis epoch length");
    }
    // Start a chain with a stabilized epoch length, a stabilized compact target and a stabilized hash rate.
    run_chain(epoch_length, compact_target, U256::from(hash_rate));
}
