"""
Inference/Generation Script for MRN
"""

import argparse
import json
import numpy as np
from pathlib import Path

from core.symbol_embedding import SymbolEmbedding
from core.modular_space import ModularSpace
from core.memory import MemoryManager
from core.sequence_generator import SequenceGenerator, BeamSearchGenerator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate sequences with trained MRN model')
    
    # Model loading
    parser.add_argument('--model-dir', type=str, required=True,
                       help='Directory containing trained model')
    parser.add_argument('--checkpoint', type=str, default='best_model.npz',
                       help='Checkpoint file name')
    
    # Generation parameters
    parser.add_argument('--prompt', type=str, default='<START>,1,+,1',
                       help='Initial sequence (comma-separated symbols)')
    parser.add_argument('--length', type=int, default=10,
                       help='Number of symbols to generate')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Sampling temperature')
    parser.add_argument('--deterministic', action='store_true',
                       help='Use deterministic (argmax) selection')
    
    # Beam search options
    parser.add_argument('--beam-search', action='store_true',
                       help='Use beam search instead of sampling')
    parser.add_argument('--beam-width', type=int, default=5,
                       help='Beam width for beam search')
    
    # Output options
    parser.add_argument('--num-samples', type=int, default=1,
                       help='Number of sequences to generate')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed generation information')
    parser.add_argument('--output-file', type=str, default=None,
                       help='Save generated sequences to file')
    
    return parser.parse_args()


def load_model(model_dir, checkpoint_name):
    """Load trained model from directory."""
    model_dir = Path(model_dir)
    
    # Load configuration
    config_path = model_dir / 'config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print("Loading model configuration...")
    print(f"  Vocabulary size: {len(config['vocabulary'])}")
    print(f"  Embedding dim: {config['embedding_dim']}")
    print(f"  Modulus: {config['modulus']}")
    print(f"  History size: {config['history_size']}")
    print(f"  Parallel window: {config['parallel_window']}")
    
    # Initialize components
    embedding = SymbolEmbedding(
        vocabulary=config['vocabulary'],
        embedding_dim=config['embedding_dim'],
        modulus=config['modulus']
    )
    
    # Load embeddings
    embedding_path = model_dir / 'embeddings.npy'
    if embedding_path.exists():
        embedding.load_embeddings(str(embedding_path))
        print(f"  Loaded embeddings from {embedding_path}")
    else:
        # Try loading from checkpoint
        checkpoint_path = model_dir / checkpoint_name
        if checkpoint_path.exists():
            checkpoint = np.load(checkpoint_path, allow_pickle=True)
            embedding.E = checkpoint['embeddings']
            print(f"  Loaded embeddings from checkpoint")
    
    # Create other components
    mod_space = ModularSpace(
        modulus=config['modulus'],
        dimension=config['embedding_dim']
    )
    
    memory = MemoryManager(
        history_size=config['history_size'],
        dimension=config['embedding_dim'],
        modulus=config['modulus']
    )
    
    generator = SequenceGenerator(
        symbol_embedding=embedding,
        modular_space=mod_space,
        memory_manager=memory,
        parallel_window_size=config['parallel_window'],
        temperature=config.get('temperature', 1.0)
    )
    
    return generator, config


def main():
    """Main inference function."""
    args = parse_args()
    
    print("=" * 80)
    print("MRN Sequence Generation")
    print("=" * 80)
    
    # Load model
    generator, config = load_model(args.model_dir, args.checkpoint)
    
    # Set temperature
    generator.set_temperature(args.temperature)
    print(f"\nGeneration settings:")
    print(f"  Temperature: {args.temperature}")
    print(f"  Deterministic: {args.deterministic}")
    print(f"  Length: {args.length}")
    
    # Parse prompt
    prompt_symbols = [s.strip() for s in args.prompt.split(',')]
    print(f"\nPrompt: {' '.join(prompt_symbols)}")
    
    # Validate prompt symbols
    for symbol in prompt_symbols:
        if symbol not in config['vocabulary']:
            print(f"Warning: Symbol '{symbol}' not in vocabulary")
            return
    
    # Generate sequences
    print("\n" + "=" * 80)
    print("Generated Sequences")
    print("=" * 80)
    
    all_sequences = []
    
    if args.beam_search:
        print(f"\nUsing beam search with width={args.beam_width}")
        beam_generator = BeamSearchGenerator(
            symbol_embedding=generator.embedding,
            modular_space=generator.mod_space,
            memory_manager=generator.memory,
            parallel_window_size=generator.K,
            temperature=args.temperature,
            beam_width=args.beam_width
        )
        
        beams = beam_generator.generate_with_beam_search(
            initial_sequence=prompt_symbols,
            length=args.length
        )
        
        for i, (sequence, log_prob) in enumerate(beams):
            print(f"\nBeam {i+1} (log_prob={log_prob:.4f}):")
            print("  " + " ".join(sequence))
            all_sequences.append(sequence)
    
    else:
        # Standard sampling
        for sample_idx in range(args.num_samples):
            print(f"\nSample {sample_idx + 1}/{args.num_samples}:")
            
            generated, debug_info = generator.generate_sequence(
                initial_sequence=prompt_symbols,
                length=args.length,
                deterministic=args.deterministic,
                return_debug=args.verbose
            )
            
            # Print full sequence
            full_sequence = prompt_symbols + generated
            print("  " + " ".join(full_sequence))
            all_sequences.append(full_sequence)
            
            # Print debug information if verbose
            if args.verbose and debug_info:
                print("\n  Generation details:")
                for t, info in enumerate(debug_info):
                    print(f"\n    Step {t+1}:")
                    print(f"      Selected: {generated[t]}")
                    
                    # Show top-3 probabilities
                    probs = info['probabilities']
                    top_indices = np.argsort(probs)[-3:][::-1]
                    print(f"      Top-3 probabilities:")
                    for idx in top_indices:
                        symbol = generator.embedding.idx_to_symbol[idx]
                        prob = probs[idx]
                        print(f"        {symbol}: {prob:.4f}")
                    
                    # Show derivation paths
                    if len(info['paths']) > 0:
                        print(f"      Derivation paths ({len(info['paths'])}):")
                        for path in info['paths'][:3]:  # Show first 3
                            alpha, beta = generator.parallel_derivation.derivation.get_weights()
                            score = path.compute_score(alpha, beta)
                            print(f"        {path.v} → {path.w} (score={score:.2f})")
    
    # Save to file if requested
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(exist_ok=True, parents=True)
        
        output_data = {
            'prompt': prompt_symbols,
            'settings': {
                'length': args.length,
                'temperature': args.temperature,
                'deterministic': args.deterministic,
                'beam_search': args.beam_search,
                'beam_width': args.beam_width if args.beam_search else None
            },
            'sequences': [' '.join(seq) for seq in all_sequences]
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n\nSaved {len(all_sequences)} sequences to {output_path}")
    
    print("\n" + "=" * 80)
    print("Generation Complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
