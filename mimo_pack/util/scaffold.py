"""
Session configuration scaffolding utility.

This module provides a command-line utility to generate default session_config.json
files for MIMO recording sessions, eliminating the tedium of manually creating
configuration files for every session directory.

Author: Drew Headley
Date: 2025-01-27
"""

import argparse
import sys
from pathlib import Path
from mimo_pack.core.session import Session


def create_session_config(session_dir, overwrite=False):
    """
    Create a default session_config.json file in the specified directory.
    
    This function wraps the Session.create_default_config() method to provide
    a convenient interface for generating session configuration files with
    sensible defaults.
    
    Parameters
    ----------
    session_dir : str or Path
        Path to the session directory where session_config.json will be created.
    overwrite : bool, optional
        Whether to overwrite an existing config file. Default is False.
        
    Returns
    -------
    Path
        Path to the created session_config.json file.
        
    Raises
    ------
    FileExistsError
        If config file exists and overwrite is False.
    """
    try:
        config_path = Session.create_default_config(session_dir, overwrite=overwrite)
        print(f"SUCCESS: Created session_config.json at: {config_path}")
        print("\nGenerated configuration contains:")
        print("  - File patterns for SpikeGLX data (AP, LFP, NIDQ)")
        print("  - Default analysis parameters (LFP, CSD, evoked responses)")
        print("  - Metadata fields (subject, probe type, date, notes)")
        print("\nNext steps:")
        print("  1. Edit the config file to match your specific file naming")
        print("  2. Update analysis parameters as needed")
        print("  3. Fill in metadata fields")
        print("  4. Use Session class in your analysis scripts")
        return config_path
        
    except FileExistsError as e:
        print(f"ERROR: {e}")
        print("Use --overwrite to replace the existing configuration.")
        return None


def init_multiple_sessions(parent_dir, pattern="*", overwrite=False):
    """
    Initialize session_config.json files for multiple session directories.
    
    This function scans a parent directory for session subdirectories and
    creates default configuration files for each one that doesn't already
    have a session_config.json file.
    
    Parameters
    ----------
    parent_dir : str or Path
        Path to the parent directory containing session directories.
    pattern : str, optional
        Glob pattern to match session directories. Default is "*".
    overwrite : bool, optional
        Whether to overwrite existing config files. Default is False.
        
    Returns
    -------
    list
        List of paths where config files were created.
    """
    parent_path = Path(parent_dir)
    
    if not parent_path.exists():
        print(f"ERROR: Parent directory {parent_path} does not exist.")
        return []
    
    session_dirs = [p for p in parent_path.glob(pattern) if p.is_dir()]
    
    if not session_dirs:
        print(f"ERROR: No directories found matching pattern '{pattern}' in {parent_path}")
        return []
    
    print(f"Found {len(session_dirs)} potential session directories:")
    
    created_configs = []
    skipped = []
    errors = []
    
    for session_dir in session_dirs:
        config_path = session_dir / 'session_config.json'
        
        if config_path.exists() and not overwrite:
            skipped.append(session_dir)
            print(f"  - {session_dir.name}: skipped (config exists)")
            continue
            
        try:
            created_config = Session.create_default_config(session_dir, overwrite=overwrite)
            created_configs.append(created_config)
            print(f"  SUCCESS: {session_dir.name}: created config")
        except Exception as e:
            errors.append((session_dir, str(e)))
            print(f"  ERROR: {session_dir.name}: error - {e}")
    
    print(f"\nSummary:")
    print(f"  Created: {len(created_configs)}")
    print(f"  Skipped: {len(skipped)}")
    print(f"  Errors:  {len(errors)}")
    
    if errors:
        print("\nErrors encountered:")
        for session_dir, error in errors:
            print(f"  - {session_dir.name}: {error}")
    
    return created_configs


def main():
    """CLI entry point for the session-config scaffold utility.

    Parses command-line arguments and generates ``session_config.json``
    files for one or more recording session directories.  Supports
    single-session and batch modes, with optional overwrite and custom
    glob patterns.
    """
    parser = argparse.ArgumentParser(
        description='Generate session_config.json files for MIMO recording sessions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create config for a single session
  python -m mimo_pack.util.scaffold /path/to/session_dir

  # Create configs for multiple sessions
  python -m mimo_pack.util.scaffold /path/to/parent_dir --batch

  # Overwrite existing configs
  python -m mimo_pack.util.scaffold /path/to/session_dir --overwrite
        
  # Use custom pattern for batch processing
  python -m mimo_pack.util.scaffold /path/to/parent_dir --batch --pattern "mouse*"
        """
    )
    
    parser.add_argument(
        'directory',
        type=str,
        help='Path to session directory (single mode) or parent directory (batch mode)'
    )
    
    parser.add_argument(
        '--batch',
        action='store_true',
        help='Batch mode: create configs for all subdirectories in the specified path'
    )
    
    parser.add_argument(
        '--pattern',
        type=str,
        default='*',
        help='Glob pattern for matching session directories in batch mode (default: "*")'
    )
    
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing session_config.json files'
    )
    
    args = parser.parse_args()
    
    if args.batch:
        print(f"Batch mode: initializing session configs in {args.directory}")
        print(f"Pattern: {args.pattern}")
        if args.overwrite:
            print("Overwrite mode: existing configs will be replaced")
        print()
        
        created_configs = init_multiple_sessions(
            args.directory, 
            pattern=args.pattern, 
            overwrite=args.overwrite
        )
        
        if created_configs:
            print(f"\nSuccessfully created {len(created_configs)} configuration files.")
        else:
            print("\nNo configuration files were created.")
            sys.exit(1)
            
    else:
        print(f"Single mode: creating session config for {args.directory}")
        if args.overwrite:
            print("Overwrite mode: existing config will be replaced")
        print()
        
        config_path = create_session_config(args.directory, overwrite=args.overwrite)
        
        if config_path is None:
            sys.exit(1)


if __name__ == '__main__':
    main()