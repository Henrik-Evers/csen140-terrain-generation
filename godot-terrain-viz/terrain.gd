extends MultiMeshInstance3D


#const TERRAIN_PATH := "res://../Data/data.npy";
const TERRAIN_PATH := "res://../attempt10/generated_2.npy";
const COLORS := { # The color of each terrain type
	'sparse_vegetation': Color(0.463, 0.506, 0.394, 1.0),
	'bare_rock': Color(0.455, 0.455, 0.455, 1.0),
	'bare_sand': Color(0.598, 0.567, 0.376, 1.0),
	'water': Color(0.0, 0.564, 0.904, 1.0),
}
const HEIGHT_SCALE := 0.1; # So it fits within an actually visible range

# A very *fun* fact: Python's integers are not fixed size!
# It changes based on the size needed to accomodate the value.
# Thankfully, our value ranges are consistent here.
# The first layer is 16 bits, and the second and third layers are 8 bits.
const TUPLE_SIZE := 6; # this assumption MIGHT NOT HOLD FOR ALL DATA! it is possible that numpy might reduce the size for data with no water regions. depends on how splitting goes.

# Global data info to avoid copying.
var data : PackedByteArray;
var dim1 : int;
var dim2 : int;
var max_height := -9999;

# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	read_terrain(TERRAIN_PATH);
	
	render_terrain();
	
	setup_camera();


func setup_camera():
	# Attribution Note: This function is modified from a template created by Google Gemini.
	# I made alterations to scaling and positioning, but the core is the output of the LLM.
	# All other components of this script are my own original work.
	
	# Create a new camera
	var cam := Camera3D.new();
	add_child(cam);
	
	# Set to orthogonal projection
	cam.projection = Camera3D.PROJECTION_ORTHOGONAL;
	
	# Calculate center of data and position the camera
	var center := Vector3(dim1 / 2.0, max_height*HEIGHT_SCALE*0.5, dim2 / 2.0);
	var offset_distance := 100;
	cam.global_position = center + Vector3(offset_distance, offset_distance + offset_distance, offset_distance);
	cam.look_at(center);
	
	# Auto-scale view size
	var grid_diagonal := sqrt(pow(dim1, 2) + pow(dim2, 2) + pow(max_height*HEIGHT_SCALE, 2));
	cam.size = grid_diagonal;


# Renders the data retrieved by read_terrain() with a colored column for each pixel.
func render_terrain() -> void:
	# Create the mesh object
	self.multimesh = MultiMesh.new();
	self.multimesh.mesh = BoxMesh.new();
	
	self.multimesh.transform_format = MultiMesh.TRANSFORM_3D;
	self.multimesh.use_colors = true;
	self.multimesh.instance_count = dim1 * dim2;
	
	# Fill the multimesh with our data
	for i in (dim1 * dim2):
		# Using x and z because those are the horizontal dimensions in Godot
		var x := i % dim1;
		@warning_ignore("integer_division") # the int division is intentional but Godot keeps complaining >:(
		var z := i / dim2;
		
		var base_index := (x * dim1 + z) * TUPLE_SIZE;
		var point := [data.decode_s16(base_index), data.decode_u16(base_index+2), data.decode_u16(base_index+4)]; # Data for this pixel
		
		# Update max_height
		if point[0] > max_height:
			max_height = point[0];
		
		# Transform this block
		var t := Transform3D();
		t.origin = Vector3(x, (point[0] * HEIGHT_SCALE) / 2.0, z);
		t.basis = Basis().scaled(Vector3(1, point[0] * HEIGHT_SCALE, 1));
		self.multimesh.set_instance_transform(i, t);
		
		# Choose the color
		var color : Color;
		match point[1]:
			10: color = COLORS['sparse_vegetation'];
			16: color = COLORS['bare_rock'];
			17: color = COLORS['bare_sand'];
			20: color = COLORS['water'];
		self.multimesh.set_instance_color(i, color);
	
	# Make a material to use the colors we set
	var mat := StandardMaterial3D.new();
	mat.vertex_color_use_as_albedo = true;
	self.multimesh.mesh.surface_set_material(0, mat);


# Reads the terrain data from a saved numpy array on the disk.
func read_terrain(path: String) -> void:
	# Open the terrain file
	var file := FileAccess.open(path, FileAccess.READ);
	if file == null:
		print("Failed to open file: ", FileAccess.get_open_error())
		return
	
	# Read data out
	var magic := file.get_buffer(6);
	print(magic); # Expect to be \x93NUMPY: [147, 78, 85, 77, 80, 89]
	var version_major := file.get_8();
	var version_minor := file.get_8();
	var header_len := file.get_16();
	print(version_major); # We are processing based on version 1.0
	print(version_minor);
	print(header_len);
	
	# Decode header data
	# The header is a dictionary written in ASCII
	var header := file.get_buffer(118).get_string_from_ascii();
	print(header);
	var shape := header.substr(header.find("'shape':")+10);
	var d1_terminator := shape.find(',');
	var d2_terminator := shape.substr(d1_terminator+2).find(',');
	var d1 := shape.substr(0, d1_terminator);
	var d2 := shape.substr(d1_terminator+2, d2_terminator);
	dim1 = d1.to_int();
	dim2 = d2.to_int();
	print(dim1);
	print(dim2);
	
	file.seek(header_len+10); # Jumps to start of data
	
	# We can keep the data as a PackedByteArray, rather than recreating the data type, and calculate addresses manually.
	# This will reduce load times and reduce memory consumption (Godot is much less performant than numpy), just be slightly more work to program.
	data = file.get_buffer(dim1 * dim2 * TUPLE_SIZE);
	
	# For full data:
	'''
	var real_dim2 := dim2; # Used when displaying only a portion of the data
	
	if (dim1 > 64):
		dim1 = 64;
	if (dim2 > 64):
		dim2 = 64;
	
	var offset := header_len + 10 + 7500000000;
	file.seek(offset); # Jump to somewhere with content, not just ocean
	for i in dim1:
		file.seek(offset + (real_dim2 * i * TUPLE_SIZE));
		data.append_array(file.get_buffer(dim2 * TUPLE_SIZE));
	'''
	
	# Index [i, j] is at data[(i * dim1 + j) * 6], as each tuple is nominally 6 bytes
	# Run decode_s16[+0], decode_u16[+2], decode_u16[+4]
	
	# Print a submatrix from the start
	for i in 8:
		for j in 8:
			var base_index := (i * dim1 + j) * TUPLE_SIZE;
			print([data.decode_s16(base_index), data.decode_u16(base_index+2), data.decode_u16(base_index+4)]);
