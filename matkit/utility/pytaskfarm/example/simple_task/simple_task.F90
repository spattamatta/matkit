program hello_world

    include 'mpif.h'
    integer :: my_world_rank
    integer :: world_size
    integer :: ierror
    integer :: pnamelen
    integer :: i
    character*(MPI_MAX_PROCESSOR_NAME) :: pname
  
    ! Initialize MPI Framework
    call MPI_INIT(ierror)
 
    ! Get the size of the world communicator
    call MPI_COMM_SIZE(MPI_COMM_WORLD, world_size, ierror)
  
    ! Get your rank under the work communicator
    call MPI_COMM_RANK(MPI_COMM_WORLD, my_world_rank, ierror)
  
    ! Get the processor name
    call MPI_GET_PROCESSOR_NAME(pname, pnamelen, ierror)

    ! Each rank print hello world along with its rank
    print*, 'Hello from rank : ', my_world_rank, "  processor name: ", pname(1:pnamelen)

    ! Close MPI
    call MPI_FINALIZE(ierror)

end program hello_world
