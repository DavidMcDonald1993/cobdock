if __name__ == "__main__":
    import sys
    import os.path
    sys.path.insert(1, 
        os.path.abspath(os.path.join(os.path.dirname(__file__), 
        os.path.pardir,
        os.path.pardir)))

import tarfile
import os.path
import shutil

def make_compressed_tarball(
    compressed_tarball_path: str, 
    source_dir: str,
    exclude: list = [],
    verbose: bool = True,
    ):
    """Generated a gzip compressed tarball located at `compressed_tarball_path`
    using the files in `source_dir`. 

    Parameters
    ----------
    compressed_tarball_path : str
        The path to the output tar.gz file.
    source_dir : str
        The directory to compress.

    Returns
    -------
    str
        The output filename of the compressed tarball.
    """
    if not compressed_tarball_path.endswith(".tar.gz"):
        compressed_tarball_path += ".tar.gz"
    if isinstance(exclude, str):
        exclude = [exclude]
    
    if verbose:
        print ("Compressing directory", source_dir, "to compressed tarball at", compressed_tarball_path)
    
    def filter_function(tarinfo):
        if any([ e in tarinfo.name for e in exclude]):
            return None
        return tarinfo
    
    with tarfile.open(compressed_tarball_path, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir), filter=filter_function)
    if verbose:
        print ("Compression complete")
    return compressed_tarball_path

def extract_compressed_tarball(
    compressed_tarball_path: str, 
    output_dir: str, 
    verbose: bool = True,
    ):
    """Extract a compresssed tarball located at `compressed_tarball_path` to directory `output_dir`.

    Parameters
    ----------
    compressed_tarball_path : str
        The filepath of the tarball.
    output_dir : str
        The directory to output the contents of the compressed tarball to.
    """
    if verbose:
        print ("Extracting compressed tarball located at", compressed_tarball_path, "to", output_dir)
    # assert compressed_tarball_path.endswith(".tar.gz")
    if not compressed_tarball_path.endswith(".tar.gz"):
        print ("Incorrect compressed tarball filename:", compressed_tarball_path)
        return None 
    with tarfile.open(compressed_tarball_path, "r:gz") as tar:
        tar.extractall(path=output_dir)
    if verbose:
        print ("Decompression complete")

def make_archive(
    archive_filename: str,
    compression_format: str,
    source_dir: str,
    verbose: bool = True,
    ):

    # if not archive_filename.endswith(compression_format):
    #     archive_filename += f".{compression_format}"

    if verbose:
        print ("Making", compression_format, "archive for directory", source_dir, )
        print ("Writing archive to", f"{archive_filename}.{compression_format}")

    archive_filename = shutil.make_archive(
        archive_filename,
        format=compression_format,
        root_dir=source_dir, # maintain internal directory structure
        base_dir=None,
    )

    return archive_filename

def make_zip_archive(
    archive_filename: str,
    source_dir: str,
    verbose: bool = True,
    ):
    return make_archive(
        archive_filename=archive_filename,
        compression_format="zip",
        source_dir=source_dir,
        verbose=verbose,
    )

if __name__ == "__main__":

    # make_compressed_tarball(
    #     compressed_tarball_path="checkme.tar.gz",
    #     source_dir="utils/io",
    #     exclude=["pose_ranking","pockets"],
    # )

    # make_zip_archive(
    #     "checkme",
    #     source_dir="utils/io/",
    #     verbose=True,
    # )

    # test gzip
    extract_compressed_tarball(
        
    )