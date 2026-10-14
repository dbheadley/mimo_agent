"""Session initialization widget for MIMO preprocessing."""

import ipywidgets as widgets
from IPython.display import display, clear_output
import os
from pathlib import Path

from mimo_pack.core.session import Session
from mimo_pack.util.scaffold import create_session_config


class SessionInitializer:
    """Interactive widget interface for session initialization"""
    
    def __init__(self):
        """Initialize the session initializer widget.

        Defaults the session directory to the current working directory
        and builds the interactive UI.
        """
        self.session_dir = Path(os.getcwd())
        self.session = None
        self.create_widgets()
    
    def create_widgets(self):
        """Build all ipywidget controls and lay out the UI."""
        # Title
        title = widgets.HTML("<h3>🎯 Interactive Session Initialization</h3>")
        
        # Session directory input
        self.dir_text = widgets.Text(
            value=str(self.session_dir),
            placeholder='Enter session directory path',
            description='Session Dir:',
            layout=widgets.Layout(width='600px')
        )
        
        # Config file status display
        self.status_html = widgets.HTML()
        
        # Configuration options
        self.create_config_checkbox = widgets.Checkbox(
            value=False,
            description='Create default config if missing',
            disabled=False
        )
        
        self.overwrite_checkbox = widgets.Checkbox(
            value=False,
            description='Overwrite existing config',
            disabled=False
        )
        
        # Control buttons
        self.check_btn = widgets.Button(
            description='🔍 Check Directory',
            button_style='info',
            layout=widgets.Layout(width='150px')
        )
        
        self.load_btn = widgets.Button(
            description='🚀 Load Session',
            button_style='success',
            layout=widgets.Layout(width='150px')
        )
        
        self.clear_btn = widgets.Button(
            description='🧹 Clear Output',
            button_style='warning',
            layout=widgets.Layout(width='150px')
        )
        
        # Output area
        self.output = widgets.Output()
        
        # Event handlers
        self.check_btn.on_click(self.on_check_directory)
        self.load_btn.on_click(self.on_load_session)
        self.clear_btn.on_click(self.on_clear_output)
        self.dir_text.observe(self.on_directory_change, names='value')
        
        # Initial check
        self.check_directory_status()
        
        # Layout
        self.container = widgets.VBox([
            title,
            widgets.HTML("<h4>📁 Session Directory:</h4>"),
            self.dir_text,
            self.status_html,
            widgets.HTML("<h4>⚙️ Configuration Options:</h4>"),
            self.create_config_checkbox,
            self.overwrite_checkbox,
            widgets.HTML("<h4>🎮 Controls:</h4>"),
            widgets.HBox([self.check_btn, self.load_btn, self.clear_btn]),
            self.output
        ])
    
    def on_directory_change(self, change):
        """Update directory and check status when path changes"""
        self.session_dir = Path(change['new'])
        self.check_directory_status()
    
    def check_directory_status(self):
        """Check the status of the session directory and config file"""
        config_path = self.session_dir / "session_config.json"
        
        if not self.session_dir.exists():
            status = "❌ Directory does not exist"
            config_status = ""
        elif not self.session_dir.is_dir():
            status = "❌ Path is not a directory"
            config_status = ""
        else:
            status = "✅ Directory exists"
            if config_path.exists():
                config_status = "<br>📋 <b>session_config.json</b> found"
            else:
                config_status = "<br>⚠️ <b>session_config.json</b> missing"
        
        self.status_html.value = f"<div style='padding: 10px; border: 1px solid #ddd;'>{status}{config_status}</div>"
    
    def on_check_directory(self, btn):
        """Scan the session directory for data files and report findings."""
        with self.output:
            print("\n" + "="*50)
            print("🔍 Checking directory...")
            self.check_directory_status()
            
            # Look for data files
            session_path = Path(self.session_dir)
            if session_path.exists():
                bin_files = list(session_path.glob("**/*.bin"))
                dclut_files = list(session_path.glob("**/*dclut.json"))
                
                print(f"📊 Found {len(bin_files)} .bin files")
                print(f"📊 Found {len(dclut_files)} dclut.json files")
                
                if bin_files:
                    print("\n📁 Sample files found:")
                    for f in bin_files[:3]:
                        print(f"  • {f.name}")
                    if len(bin_files) > 3:
                        print(f"  ... and {len(bin_files) - 3} more")
            
            print("✅ Directory check complete")
    
    def on_load_session(self, btn):
        """Create or load the Session object from the configured directory."""
        with self.output:
            print("\n" + "="*50)
            print("🚀 Loading session...")
            
            session_dir = Path(self.dir_text.value)
            config_path = session_dir / "session_config.json"
            
            try:
                # Handle config creation if needed
                if not config_path.exists() and self.create_config_checkbox.value:
                    print("📋 Creating default configuration...")
                    config_path = create_session_config(session_dir, overwrite=False)
                    print(f"✅ Created config at: {config_path}")
                elif self.overwrite_checkbox.value:
                    print("📋 Creating/overwriting configuration...")
                    config_path = create_session_config(session_dir, overwrite=True)
                    print(f"✅ Updated config at: {config_path}")
                elif not config_path.exists():
                    print("❌ No session_config.json found.")
                    print("💡 Enable 'Create default config' option to create one automatically.")
                    return
                
                # Load the session
                self.session = Session(session_dir)
                print(f"✅ Session loaded successfully!")
                print(f"   Directory: {self.session.dir}")
                
                # Display all session object fields
                print("\n📦 Session Object Fields:")
                for attr in dir(self.session):
                    if not attr.startswith('_'):  # Skip private/dunder attributes
                        try:
                            val = getattr(self.session, attr)
                            if not callable(val):  # Only show non-method attributes
                                val_str = str(val)
                                if len(val_str) > 80:
                                    val_str = val_str[:77] + "..."
                                print(f"   • {attr}: {val_str}")
                        except Exception:
                            pass
                
                # Show configuration summary
                if hasattr(self.session, 'config'):
                    print(f"\n📋 Configuration Details:")
                    if 'paths' in self.session.config:
                        print(f"   Paths ({len(self.session.config['paths'])} patterns):")
                        for key, pattern in self.session.config['paths'].items():
                            print(f"      • {key}: {pattern}")
                    if 'parameters' in self.session.config:
                        print(f"   Parameters ({len(self.session.config['parameters'])} sections):")
                        for key in self.session.config['parameters'].keys():
                            print(f"      • {key}")
                    if 'metadata' in self.session.config:
                        print(f"   Metadata:")
                        for key, val in self.session.config['metadata'].items():
                            print(f"      • {key}: {val}")
                
                print("\n🎯 Session ready for preprocessing!")
                
            except Exception as e:
                print(f"❌ Failed to load session: {e}")
                print("💡 Check your directory path and ensure it contains valid session data")
    
    def on_clear_output(self, btn):
        """Clear the output area"""
        with self.output:
            clear_output()
    
    def display(self):
        """Render the widget in the current notebook cell."""
        display(self.container)
