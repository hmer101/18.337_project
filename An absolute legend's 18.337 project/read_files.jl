using DelimitedFiles

# Walk through the specified directory (and all subdirectories) to find data files
function find_files(root, exclude = [], extension = "tsv")
    # root : directory to start search from
    # exclude : iterable of strings by which to reject files or directories
    # extension : file extension of choice, without the .
    # This function won't follow symbolic links

    # We'll need a size-changing vector because it's unclear how many data files to expect
    fnames = Vector{String}(undef, 0)

    # Find the files by walking the directory tree
    for (root_, dirs_, files_) in walkdir(root)

        # The above loop goes through all the subdirectories starting at root. Now go through all the files in the subdirectory.
        for file in files_

            # Honor the exclusions
            keep_going = true
            for bad in exclude
                if occursin(bad, file) || occursin(bad, root_)
                    keep_going = false
                    break
                end
            end

            # If it has the proper extension and it's not excluded...
            #println(file)
            #println(file[end-length(extension):end])
            #println("." * extension)
            if keep_going && file[end-length(extension):end] == "." * extension
                #println("~~~ I'm inside")
                push!(fnames, joinpath(root_, file))
            end

        end

    end

    return fnames

end


# Define a type that will hold our data, row labels, and column labels
struct DataFile{T}
    fname :: String
    data :: Array{T, 2}
    column_labels :: Array{String, 2}
    row_labels :: Array{String, 2}
end


# Read the data from a single file
function read_data(file, delimiter, column_label_lines, row_label_lines, data_type, trim_data_by)
    # file : the full path to the file of interest
    # delimiter : delimiter used within the data file, assuming a simple delimited file (e.g. csv, tsv)
    # column_label_lines and row_label_lines : number of lines to exclude from top and left of file, respectively, to isolate the data
    # File is assumed to be formatted as a simple table of homogeneous data type (excepting row and column headers)
    # trim_data_by : vector of two tuples, where the data are trimmed from the start or end by [(t_start, t_end), (f_start, f_end)]. Note t_end and f_end are measured FROM THE END, not from 1.

    # One allocation: raw data read in from file
    file_data = DelimitedFiles.readdlm(file, delimiter) # May fail if the newline character is \r\n rather than \n

    # Separate out the main data
    # Second allocation: a separate array for numeric data of uniform type
    data = Array{data_type, 2}(@view file_data[1+column_label_lines+trim_data_by[1][1]:end-trim_data_by[1][2], 1+row_label_lines+trim_data_by[2][1]:end-trim_data_by[2][2]]) # I think the view is necessary to avoid allocations in the slicing
    # TODO couldn't I just do the slice? Should be type-stable. Or, hm....maybe not, since it's an Any array originally....

    # Record the row and column labels too
    # Since file_data will be a large multi-type array, it's better to slice new sub-arrays for the smaller lists as well (so file_data can be garbage collected)

    # Column labels
    if column_label_lines > 0
        top_unused = string.(file_data[1:1+column_label_lines-1, 1+trim_data_by[2][1]:end-trim_data_by[2][2]])#Vector{String}()
    else
        #top_unused = nothing
        top_unused = Array{String}(undef, 0, 0)
    end

    # Row labels
    if row_label_lines > 0
        left_unused = string.(file_data[1+trim_data_by[1][1]:end-trim_data_by[1][2], 1:1+row_label_lines-1])
    else
        #left_unused = nothing
        left_unused = Array{String}(undef, 0, 0)
    end

    # Return a handy data type
    return DataFile(basename(file), data, top_unused, left_unused)

end


# Read the data from a collection of files, and arrange the data in a useful structure
function assemble_data(files; grouping = 2, exclude = [], delimiter = '\t', column_label_lines = 1, row_label_lines = 1, data_type = Float64, trim_data_by = [(0, 0), (0, 0)])
    # files : a list of full file paths of interest
    # grouping : how many files to group together
    # exclude : iterable of strings by which to reject files or directories
    # Assumes the files are sorted alphabetically, in order

    # It's unknown how many files will be excluded, so this will also be variable-size
    data = Vector{Vector{DataFile}}()
    f = []

    i = 1
    this_data = nothing
    for file in files

        # Honor the exclusions
        keep_going = true
        for bad in exclude
            if occursin(bad, file)
                keep_going = false
                break
            end
        end

        # Read and organize the data
        if keep_going
            # Proceed as normal

            if i == 1
                this_data = Vector{DataFile}(undef, grouping)
            end

            this_data[i] = read_data(file, delimiter, column_label_lines, row_label_lines, data_type, trim_data_by)

            if i == grouping
                i = 1
                push!(data, this_data)
            else
                i += 1
            end

        else
            # Make an exception for certain files
            # These aren't the usual file, but might be something like "frequencies" or "settings"
            if occursin("frequencies", file)
                # In this case, "frequencies"
                f = vec(read_data(file, delimiter, 1, 0, data_type, [(0, 0), trim_data_by[2]]).data)
                    # Note that the trim_data_by in the second position is necessary; that accomplishes any requested frequency trimming
            end
        end

    end

    return data, f

end
