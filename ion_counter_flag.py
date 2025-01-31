import MDAnalysis as mda
import numpy as np

class IonPermeationAnalysis:
    def __init__(self, 
                 universe: mda.Universe,
                 ion_selection: str,
                 upper_gate_residues: list,
                 lower_gate_residues: list,
                 channel_radius: float = 8.0):
        
        self.u = universe
        self.ion_selection = ion_selection
        self.channel_radius = channel_radius
        
        # Gate definitions
        self.upper_gate_residues = upper_gate_residues
        self.lower_gate_residues = lower_gate_residues
        
        # Tracking
        self.ion_states = {}  # Track ion positions
        self.permeation_events = []
        
    def _get_gate_centers(self, ts):
        """Get centers of mass for both gates."""
        upper_sel = self.u.select_atoms(f"resid {' '.join(map(str, self.upper_gate_residues))}")
        lower_sel = self.u.select_atoms(f"resid {' '.join(map(str, self.lower_gate_residues))}")
        
        upper_center = upper_sel.center_of_mass()
        lower_center = lower_sel.center_of_mass()
        
        # Calculate channel axis and center
        self.channel_vector = lower_center - upper_center
        self.channel_length = np.linalg.norm(self.channel_vector)
        self.channel_axis = self.channel_vector / self.channel_length
        self.channel_center = (upper_center + lower_center) / 2
        
        print(f"Frame {ts.frame}: Computed gate centers. Upper: {upper_center}, Lower: {lower_center}")
        return upper_center, lower_center

    def _is_within_cylinder(self, point):
        """Check if point is within the dynamically aligned cylindrical channel region."""
        v = point - self.channel_center
        proj = np.dot(v, self.channel_axis) * self.channel_axis
        radial = v - proj
        radial_dist = np.linalg.norm(radial)
        axial_pos = np.dot(v, self.channel_axis)
        
        return (radial_dist <= self.channel_radius) and (-self.channel_length/2 <= axial_pos <= self.channel_length/2)
    
    def analyze(self):
        """Main function to track ion permeation dynamically."""
        ions = self.u.select_atoms(self.ion_selection)
        
        for ts in self.u.trajectory:
            print(f"Frame {ts.frame}: Processing frame...")
            upper_center, lower_center = self._get_gate_centers(ts)
            
            for ion in ions:
                resid = ion.resid
                ion_pos = ion.position
                
                if resid not in self.ion_states:
                    self.ion_states[resid] = {'flag': (0, 0)}
                
                in_cylinder = self._is_within_cylinder(ion_pos)
                ion_vec = ion_pos - self.channel_center
                ion_z = np.dot(ion_vec, self.channel_axis)
                
                upper_z = np.dot(upper_center - self.channel_center, self.channel_axis)
                lower_z = np.dot(lower_center - self.channel_center, self.channel_axis)
                
                print(f"Frame {ts.frame}: Ion {resid} at position {ion_pos}, flag {self.ion_states[resid]['flag']}, Z-pos {ion_z:.2f}")
                
                # Handle entering from upper gate
                if self.ion_states[resid]['flag'] == (0, 0) and ion_z > upper_z:
                    self.ion_states[resid]['flag'] = (0, 1)
                    print(f"Frame {ts.frame}: Ion {resid} entered channel (0,0 → 0,1)")

                # Handle ion exiting back through the upper gate (reset)
                elif self.ion_states[resid]['flag'] == (0, 1) and ion_z > upper_z:
                    self.ion_states[resid]['flag'] = (0, 0)
                    print(f"Frame {ts.frame}: Ion {resid} exited back through upper gate (0,1 → 0,0)")

                # Handle crossing lower gate (true permeation)
                elif self.ion_states[resid]['flag'] == (0, 1) and ion_z < lower_z:
                    self.ion_states[resid]['flag'] = (1, 1)
                    self.permeation_events.append({'ion_id': resid, 'frame': ts.frame})
                    print(f"Frame {ts.frame}: Ion {resid} permeated (0,1 → 1,1)")

                # Handle ion reversing after permeation (back to 0,1)
                elif self.ion_states[resid]['flag'] == (1,1) and ion_z > lower_z:
                    self.ion_states[resid]['flag'] = (0,1)
                    print(f"Frame {ts.frame}: Ion {resid} reversed back above lower gate (1,1 → 0,1)")
        
        print("\nFinal Permeation Events:")
        for event in self.permeation_events:
            print(f"Ion {event['ion_id']} permeated at frame {event['frame']}")

# Load simulation data
topology_file = "your_topology.pdb"  # Replace with your topology file (.pdb or .gro)
trajectory_file = "your_trajectory.xtc"  # Replace with your trajectory file (.xtc or .trr)

u = mda.Universe(topology_file, trajectory_file)

# Define ion selection (example for K+ ions)
ion_selection = "resname K+"

# Define the gate residues (replace with actual residue IDs)
upper_gate_residues = [138, 788, 463, 1113]
lower_gate_residues = list(range(259, 266)) + list(range(909, 916)) + list(range(584, 591)) + list(range(1234, 1241))

# Initialize the analysis
analysis = IonPermeationAnalysis(
    universe=u,
    ion_selection=ion_selection,
    upper_gate_residues=upper_gate_residues,
    lower_gate_residues=lower_gate_residues,
    channel_radius=8.0  # Adjust based on your system
)

# Run the analysis
analysis.analyze()
