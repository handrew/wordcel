"""Auto-completion functions for the Wordcel CLI."""
import os
import click
from pathlib import Path
from typing import List, Tuple, Optional


def complete_pipeline_files(ctx, param, incomplete):
    """Auto-complete pipeline YAML files in current directory."""
    try:
        files = []
        for f in os.listdir('.'):
            if f.endswith(('.yaml', '.yml')) and f.startswith(incomplete):
                files.append(f)
        return files
    except (OSError, PermissionError):
        return []


def complete_python_files(ctx, param, incomplete):
    """Auto-complete Python files for custom nodes/functions."""
    try:
        files = []
        for f in os.listdir('.'):
            if f.endswith('.py') and f.startswith(incomplete):
                files.append(f)
        return files
    except (OSError, PermissionError):
        return []


def complete_node_types(ctx, param, incomplete):
    """Auto-complete available node types."""
    try:
        from wordcel.dag.nodes import NODE_TYPES
        return [
            node_type for node_type in NODE_TYPES.keys()
            if node_type.startswith(incomplete)
        ]
    except ImportError:
        return []


def complete_log_levels(ctx, param, incomplete):
    """Auto-complete log levels."""
    levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR']
    return [level for level in levels if level.startswith(incomplete.upper())]


def complete_output_formats(ctx, param, incomplete):
    """Auto-complete visualization output formats."""
    formats = ['png', 'jpg', 'jpeg', 'svg', 'pdf']
    return [fmt for fmt in formats if fmt.startswith(incomplete.lower())]


def complete_templates(ctx, param, incomplete):
    """Auto-complete pipeline templates."""
    templates = ['basic', 'advanced', 'rag', 'analysis', 'data-pipeline', 'sentiment']
    return [template for template in templates if template.startswith(incomplete)]


def complete_backends(ctx, param, incomplete):
    """Auto-complete backend types."""
    try:
        from wordcel.dag.backends import BackendRegistry
        backends = list(BackendRegistry._registry.keys())
        return [backend for backend in backends if backend.startswith(incomplete)]
    except ImportError:
        return ['local', 'memory']


def complete_config_params(ctx, param, incomplete):
    """Auto-complete common config parameter names."""
    common_params = [
        'model', 'temperature', 'max_tokens', 'api_key', 
        'base_url', 'timeout', 'retries', 'cache_dir'
    ]
    return [param for param in common_params if param.startswith(incomplete)]


def complete_existing_files(ctx, param, incomplete):
    """Auto-complete existing files and directories."""
    try:
        path = Path(incomplete)
        if path.is_dir():
            # Complete files in directory
            return [
                str(path / f.name) for f in path.iterdir()
                if f.name.startswith('')
            ]
        else:
            # Complete in parent directory
            parent = path.parent if path.parent != path else Path('.')
            prefix = path.name
            return [
                str(parent / f.name) for f in parent.iterdir()
                if f.name.startswith(prefix)
            ]
    except (OSError, PermissionError):
        return []


def complete_image_files(ctx, param, incomplete):
    """Auto-complete image files for visualization output."""
    try:
        path = Path(incomplete)
        if path.suffix:
            # Has extension, complete as-is
            return [incomplete] if incomplete.endswith(('.png', '.jpg', '.jpeg', '.svg', '.pdf')) else []
        else:
            # No extension, suggest with extensions
            return [
                f"{incomplete}.png",
                f"{incomplete}.jpg", 
                f"{incomplete}.svg",
                f"{incomplete}.pdf"
            ]
    except:
        return []


# Completion installation functions
def get_completion_script(shell: str) -> str:
    """Generate completion script for specified shell."""
    if shell == 'bash':
        return get_bash_completion_script()
    elif shell == 'zsh':
        return get_zsh_completion_script()
    elif shell == 'fish':
        return get_fish_completion_script()
    else:
        raise ValueError(f"Unsupported shell: {shell}")


def get_bash_completion_script() -> str:
    """Generate bash completion script."""
    return '''
# Wordcel bash completion
_wordcel_completion() {
    local IFS=$'\\n'
    local response
    
    response=$(env COMP_WORDS="${COMP_WORDS[*]}" COMP_CWORD=$COMP_CWORD _WORDCEL_COMPLETE=bash_complete $1)
    
    for completion in $response; do
        IFS=',' read type value <<< "$completion"
        
        if [[ $type == 'dir' ]]; then
            COMPREPLY=()
            compopt -o dirnames
        elif [[ $type == 'file' ]]; then
            COMPREPLY=()
            compopt -o default
        elif [[ $type == 'plain' ]]; then
            COMPREPLY+=($value)
        fi
    done
    
    return 0
}

complete -o nosort -F _wordcel_completion wordcel
'''


def get_zsh_completion_script() -> str:
    """Generate zsh completion script."""
    return '''
# Wordcel zsh completion
#compdef wordcel

_wordcel() {
    local -a completions
    local -a completions_with_descriptions
    local -a response
    (( ! $+commands[wordcel] )) && return 1

    response=("${(@f)$(_WORDCEL_COMPLETE=zsh_complete wordcel "${words[@]}")}")

    for type_and_completion in $response; do
        completions_with_descriptions+=("$type_and_completion")
    done

    if [[ -n $completions_with_descriptions ]]; then
        _describe -V unsorted completions_with_descriptions -U
    fi
}

compdef _wordcel wordcel
'''


def get_fish_completion_script() -> str:
    """Generate fish completion script."""
    return '''
# Wordcel fish completion
function __fish_wordcel_complete
    set -lx _WORDCEL_COMPLETE fish_complete
    set -lx COMP_WORDS (commandline -opc) (commandline -ct)
    wordcel
end

complete --no-files --command wordcel --arguments '(__fish_wordcel_complete)'
'''


def install_completion(shell: Optional[str] = None) -> Tuple[bool, str]:
    """Install shell completion for wordcel.
    
    Args:
        shell: Shell type (bash, zsh, fish). Auto-detected if None.
        
    Returns:
        Tuple of (success, message)
    """
    if not shell:
        shell = detect_shell()
    
    if not shell:
        return False, "Could not detect shell type. Please specify --shell"
    
    try:
        script = get_completion_script(shell)
        success, path = write_completion_script(shell, script)
        
        if success:
            return True, f"✅ {shell.title()} completion installed to {path}"
        else:
            return False, f"❌ Failed to install completion: {path}"
            
    except ValueError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Unexpected error: {e}"


def detect_shell() -> Optional[str]:
    """Detect the current shell."""
    shell_path = os.environ.get('SHELL', '')
    if 'bash' in shell_path:
        return 'bash'
    elif 'zsh' in shell_path:
        return 'zsh'
    elif 'fish' in shell_path:
        return 'fish'
    return None


def write_completion_script(shell: str, script: str) -> Tuple[bool, str]:
    """Write completion script to appropriate location.
    
    Returns:
        Tuple of (success, path_or_error_message)
    """
    try:
        if shell == 'bash':
            # Try multiple locations
            locations = [
                Path.home() / '.bash_completion.d' / 'wordcel',
                Path.home() / '.local/share/bash-completion/completions/wordcel',
                Path('/usr/local/etc/bash_completion.d/wordcel'),
            ]
            
            for location in locations:
                try:
                    location.parent.mkdir(parents=True, exist_ok=True)
                    location.write_text(script)
                    return True, str(location)
                except PermissionError:
                    continue
                    
        elif shell == 'zsh':
            # Try multiple locations
            locations = [
                Path.home() / '.zsh/completions/_wordcel',
                Path.home() / '.local/share/zsh/site-functions/_wordcel',
            ]
            
            for location in locations:
                try:
                    location.parent.mkdir(parents=True, exist_ok=True)
                    location.write_text(script)
                    return True, str(location)
                except PermissionError:
                    continue
                    
        elif shell == 'fish':
            location = Path.home() / '.config/fish/completions/wordcel.fish'
            location.parent.mkdir(parents=True, exist_ok=True)
            location.write_text(script)
            return True, str(location)
            
        return False, "No writable completion directory found"
        
    except Exception as e:
        return False, str(e)