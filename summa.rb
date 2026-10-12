File.readable?("test.txt")
File.writable?("test.txt")
File.executable?("test.txt")

File.zero?("test.txt")
File.size?("test.txt")

File::ftype("test.txt")
File::ctime("test.txt")
File::mtime("test.txt")
File::atime("test.txt")

Dir.chdir("C:\\Users\\sriti\\Downloads\\test.txt")
puts Dir.pwd

Dir.chdir("")
Dir.foreach("/usr/bin") do |entry|
    puts entry
end

# creating a directory
Dir.mkdir("mynew")

# deleting a directory
Dir.delete("testdir")


f = Tempfile.new("tingtong")